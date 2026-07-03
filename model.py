import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Dropout, ELU, LayerNorm, Linear, ModuleList, Parameter, Sequential


class FocalLoss(torch.nn.Module):
    """Focal loss for binary classification"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class FusionPredictor(torch.nn.Module):
    """
    Graph Attention Network for gene fusion edge classification.

      - Edge encoder MLP: encodes the chimeric junction evidence.
      - Node encoder: encodes the node features (TPM, length, optional Mitelman tags).
      - GATv2Conv with 2 default layers and a small number of heads (default 2).
      - Edge classifier MLP that scores each directed edge using donor + acceptor GAT embeddings.

    Inputs (per forward pass):
      x: (N, num_node_features)
      edge_index: (2, E)
      edge_attr: (E, num_edge_features)

    Output:
      logits: (E,) raw scores, apply sigmoid externally for probabilities
    """

    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, dropout=0.3, heads=2, edge_embed_dim=32, prior_pos_rate=None, num_gnn_layers=2, node_skip=True):
        super().__init__()
        self.prior_pos_rate = prior_pos_rate
        self.num_gnn_layers = num_gnn_layers
        self.node_skip = node_skip
        self.hidden_dim = hidden_dim
        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be >= 1")

        self.edge_encoder = Sequential(
            Linear(num_edge_features, edge_embed_dim),
            LayerNorm(edge_embed_dim),
            ELU(),
            Dropout(p=dropout),
            Linear(edge_embed_dim, edge_embed_dim),
            LayerNorm(edge_embed_dim),
            ELU(),
        )

        self.node_encoder = Sequential(
            Linear(num_node_features, hidden_dim),
            LayerNorm(hidden_dim),
            ELU(),
        )
        # -------------------------------------------------------------------------
        # Graph attention layers
        # -------------------------------------------------------------------------

        self.convs = ModuleList()
        self.norms = ModuleList()
        in_dim = hidden_dim
        for i in range(num_gnn_layers):
            is_last = i == num_gnn_layers - 1
            self.convs.append(
                GATv2Conv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    concat=(not is_last),
                    dropout=dropout,
                    edge_dim=edge_embed_dim,
                )
            )
            out_dim = hidden_dim * heads if not is_last else hidden_dim
            self.norms.append(LayerNorm(out_dim))
            in_dim = out_dim

        first_out = hidden_dim * heads if num_gnn_layers > 1 else hidden_dim
        self.res_first = Linear(hidden_dim, first_out, bias=False)

        self.dropout = Dropout(p=dropout)

        # -------------------------------------------------------------------------
        # Edge classifier
        # -------------------------------------------------------------------------
        edge_input_dim = (hidden_dim * 2) + edge_embed_dim
        if node_skip:
            edge_input_dim += hidden_dim * 2
        self.edge_mlp = Sequential(
            Linear(edge_input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ELU(),
            Dropout(p=dropout),
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            ELU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, 1),
        )

        if prior_pos_rate is not None and 0.0 < prior_pos_rate < 1.0:
            bias_value = math.log(prior_pos_rate / (1.0 - prior_pos_rate))
            self.edge_mlp[-1].bias.data.fill_(bias_value)

        # The gate is a single learnable scalar, initialised so that softplus(gate) ≈ 0.5 (log(e - 1) = -0.541) at the start of training.
        self.evidence_gate = Parameter(torch.tensor(-0.541, dtype=torch.float32))

    def forward(self, x, edge_index, edge_attr, edge_evidence=None):
        edge_embed = self.edge_encoder(edge_attr)
        node_h = self.node_encoder(x)

        h = node_h
        for i, (conv, ln) in enumerate(zip(self.convs, self.norms)):
            out = F.elu(ln(conv(h, edge_index, edge_embed)))
            if i == 0:
                out = out + self.res_first(node_h)
            else:
                if out.shape == h.shape:
                    out = out + h
            h = self.dropout(out)

        src = h[edge_index[0]]
        dst = h[edge_index[1]]

        if self.node_skip:
            src_skip = node_h[edge_index[0]]
            dst_skip = node_h[edge_index[1]]
            cls_in = torch.cat([src, dst, src_skip, dst_skip, edge_embed], dim=1)
        else:
            cls_in = torch.cat([src, dst, edge_embed], dim=1)
        logits = self.edge_mlp(cls_in).squeeze(-1)

        if edge_evidence is not None and edge_evidence.numel() > 0:
            ev = edge_evidence.view(-1).to(logits.dtype)
            logits = logits + F.softplus(self.evidence_gate) * ev
        return logits

    @staticmethod
    def _subsample_for_loss(logits, y, neg_ratio):
        """Keep all positives and a mix of hard + random negatives for loss"""
        pos_mask = y == 1.0
        num_pos = pos_mask.sum().item()
        if num_pos == 0 or neg_ratio is None:
            return logits, y
        neg_indices = (~pos_mask).nonzero(as_tuple=True)[0]
        max_neg = int(num_pos * neg_ratio)
        if len(neg_indices) <= max_neg:
            return logits, y

        # Split budget: 50% hard negatives, 50% random
        n_hard = max_neg // 2
        n_random = max_neg - n_hard

        # Hard negatives: highest predicted probability among negatives
        neg_logits = logits[neg_indices].detach().squeeze()
        _, hard_order = torch.sort(neg_logits, descending=True)
        hard_idx = neg_indices[hard_order[:n_hard]]

        # Random negatives: sampled from the remaining negatives
        remaining_mask = torch.ones(len(neg_indices), dtype=torch.bool, device=neg_indices.device)
        remaining_mask[hard_order[:n_hard]] = False
        remaining = neg_indices[remaining_mask]
        if len(remaining) > n_random:
            perm = torch.randperm(len(remaining), device=remaining.device)[:n_random]
            random_idx = remaining[perm]
        else:
            random_idx = remaining

        keep = torch.cat([pos_mask.nonzero(as_tuple=True)[0], hard_idx, random_idx])
        return logits[keep], y[keep]

    @staticmethod
    def _augment_positive_edges(edge_attr, y, noise_std, continuous_cols=None):
        """Add small Gaussian noise to continuous edge features of positive edges"""
        if noise_std <= 0 or y.sum() == 0:
            return edge_attr
        if continuous_cols is None:
            continuous_cols = list(range(min(5, edge_attr.shape[1])))
        noisy = edge_attr.clone()
        pos_idx = y.nonzero(as_tuple=True)[0]
        cols_t = torch.tensor(continuous_cols, device=edge_attr.device)
        noise = torch.randn(len(pos_idx), len(cols_t), device=edge_attr.device) * noise_std
        noisy[pos_idx[:, None], cols_t[None, :]] += noise
        return noisy

    @torch.no_grad()
    def _eval_f1(self, dataset, device=None, return_details=False):
        """Calculate classification metrics"""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        all_logits, all_targets = [], []
        for data in dataset:
            data = data.to(device)
            logits = self.forward(
                data.x,
                data.edge_index,
                data.edge_attr,
                edge_evidence=getattr(data, "edge_evidence", None),
            )
            all_logits.append(logits.cpu())
            all_targets.append(data.y.cpu())
        self.train()
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        probs = torch.sigmoid(logits)

        f1_at_05, _, _ = _f1_at_threshold(probs, targets, 0.5)
        f1_best, best_thr, precision_best, recall_best, auprc = _f1_best_threshold(probs, targets)

        if not return_details:
            return f1_best

        return {
            "f1_at_05": f1_at_05,
            "f1_best": f1_best,
            "best_thr": best_thr,
            "precision": precision_best,
            "recall": recall_best,
            "auprc": auprc,
            "num_pos": int(targets.sum().item()),
            "num_total": int(targets.numel()),
        }

    # ------------------------------------------------------------------
    # Single-graph training
    # ------------------------------------------------------------------
    def train_model(self, graph_data, epochs=100, lr=0.01, patience=15, optimizer_state_dict=None, neg_ratio=50, use_focal_loss=False, focal_gamma=2.0):
        """Train on a single graph"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        num_pos = int(graph_data.y.sum().item())
        num_neg = graph_data.y.shape[0] - num_pos

        if use_focal_loss:
            criterion = FocalLoss(alpha=0.75, gamma=focal_gamma)
            print(f"[*] positives: {num_pos} | negatives: {num_neg} | loss: focal (gamma={focal_gamma})")
        else:
            pw = min(num_neg / max(num_pos, 1), 100.0)
            pos_weight = torch.tensor([pw])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[*] positives: {num_pos} | negatives: {num_neg} | pos_weight: {pos_weight.item():.1f} (capped at 100)")

        if neg_ratio is not None:
            effective_neg = min(num_neg, int(num_pos * neg_ratio))
            print(f"[*] Negative subsampling: ~{effective_neg} negatives per epoch (ratio {neg_ratio}:1)")

        self.train()
        best_loss, epochs_no_improve, last_epoch = float("inf"), 0, 0
        best_state_dict = None

        for epoch in range(epochs):
            last_epoch = epoch + 1
            optimizer.zero_grad()
            logits = self.forward(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                edge_evidence=getattr(graph_data, "edge_evidence", None),
            )

            logits_sub, y_sub = self._subsample_for_loss(logits, graph_data.y, neg_ratio)
            loss = criterion(logits_sub, y_sub.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                self._log_metrics(logits, graph_data.y, loss.item(), epoch + 1, epochs, optimizer)

            if loss.item() < best_loss:
                best_loss, epochs_no_improve = loss.item(), 0
                best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                print(f"[*] New best model at epoch {epoch + 1} — loss: {best_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[*] Early stopping at epoch {epoch + 1}.")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print(f"[*] Restored best model weights (loss: {best_loss:.4f}).")

        print("Training complete.")
        return {"optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss, "epochs_ran": last_epoch}

    # ------------------------------------------------------------------
    # Multi-graph training
    # ------------------------------------------------------------------

    def train_model_multi_graph(
        self,
        dataset,
        val_dataset=None,
        epochs=200,
        lr=0.005,
        patience=20,
        batch_size=4,
        optimizer_state_dict=None,
        dataloader_workers=0,
        neg_ratio=50,
        use_focal_loss=True,
        focal_gamma=2.0,
        feature_noise_std=0.01,
        select_metric="auprc",
    ):
        """Train on a list of graphs using mini-batch gradient descent"""
        from torch_geometric.loader import DataLoader

        # Detect the best available device.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"[*] Training device: {device}")
        self.to(device)

        if use_focal_loss and neg_ratio is not None:
            print(f"[!] Disabling negative subsampling (neg_ratio={neg_ratio}) — " "focal loss handles class imbalance natively.")
            neg_ratio = None

        if dataloader_workers > 0:
            print(f"[!] dataloader_workers={dataloader_workers} ignored — " "using 0 to avoid shared-memory crashes with in-memory PyG graphs.")
            dataloader_workers = 0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        _nef = dataset[0].edge_attr.shape[1] if len(dataset) > 0 else 15
        _continuous_edge_cols = [0, 1, 2, 3, 4, 13, 14, 17, 18, 19, 20, 21, 22]
        if _nef == 31:  # with Mitelman
            _continuous_edge_cols += [25, 26, 27, 28, 29, 30]
        else:
            _continuous_edge_cols += [24, 25, 26, 27, 28]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        all_labels = torch.cat([data.y for data in dataset])
        num_pos = int(all_labels.sum().item())
        num_neg = all_labels.shape[0] - num_pos

        if use_focal_loss:
            criterion = FocalLoss(alpha=0.75, gamma=focal_gamma).to(device)
            print(f"[*] graphs: {len(dataset)} | positives: {num_pos} | " f"negatives: {num_neg} | loss: focal (gamma={focal_gamma})")
        else:
            pw = min(num_neg / max(num_pos, 1), 100.0)
            pos_weight = torch.tensor([pw], device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[*] graphs: {len(dataset)} | positives: {num_pos} | " f"negatives: {num_neg} | pos_weight: {pos_weight.item():.1f} (capped at 100)")

        if neg_ratio is not None:
            effective_neg = min(num_neg, int(num_pos * neg_ratio))
            print(f"[*] Negative subsampling: ~{effective_neg} negatives per batch (ratio {neg_ratio}:1)")

        self.train()
        best_loss = float("inf")
        best_val_f1, best_train_f1 = -1.0, -1.0
        best_val_thr, best_val_auprc = 0.5, 0.0
        best_state_dict, epochs_no_improve, last_epoch = None, 0, 0

        for epoch in range(epochs):
            last_epoch = epoch + 1
            epoch_loss = 0.0
            all_logits = []
            all_targets = []

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                edge_attr = self._augment_positive_edges(batch.edge_attr, batch.y, feature_noise_std, _continuous_edge_cols)
                logits = self.forward(
                    batch.x,
                    batch.edge_index,
                    edge_attr,
                    edge_evidence=getattr(batch, "edge_evidence", None),
                )

                logits_sub, y_sub = self._subsample_for_loss(logits, batch.y, neg_ratio)
                loss = criterion(logits_sub, y_sub.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_targets.append(batch.y.detach().cpu())

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            val_metrics = None
            if val_dataset is not None:
                val_metrics = self._eval_f1(val_dataset, device, return_details=True)

            if (epoch + 1) % 10 == 0:
                combined_logits = torch.cat(all_logits)
                combined_targets = torch.cat(all_targets)
                self._log_metrics(combined_logits, combined_targets, avg_loss, epoch + 1, epochs, optimizer)
                if val_metrics is not None:
                    print(
                        f"  Val F1@0.5: {val_metrics['f1_at_05']:.4f} | "
                        f"F1@best: {val_metrics['f1_best']:.4f}@thr={val_metrics['best_thr']:.3f} "
                        f"(P={val_metrics['precision']:.2f} R={val_metrics['recall']:.2f}) | "
                        f"AUPRC: {val_metrics['auprc']:.4f} | "
                        f"positives: {val_metrics['num_pos']}/{val_metrics['num_total']}"
                    )

            if val_metrics is not None:
                if select_metric == "auprc":
                    val_score = val_metrics["auprc"]
                    _score_label = "AUPRC"
                else:
                    val_score = val_metrics["f1_best"]
                    _score_label = "F1"

                if val_score > best_val_f1:
                    best_val_f1 = val_score
                    best_val_thr = val_metrics["best_thr"]
                    best_val_auprc = val_metrics["auprc"]
                    best_train_f1 = self._eval_f1(dataset, device)
                    best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                    print(
                        f"[*] New best model at epoch {epoch + 1} "
                        f"— val {_score_label}: {val_score:.4f} | "
                        f"val F1: {val_metrics['f1_best']:.4f}@thr={best_val_thr:.3f} | "
                        f"train F1: {best_train_f1:.4f} | "
                        f"val AUPRC: {best_val_auprc:.4f}"
                    )
                elif val_score == best_val_f1:
                    current_train_f1 = self._eval_f1(dataset, device)
                    if current_train_f1 > best_train_f1:
                        best_train_f1 = current_train_f1
                        best_val_thr = val_metrics["best_thr"]
                        best_val_auprc = val_metrics["auprc"]
                        best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                        print(
                            f"[*] New best model at epoch {epoch + 1} "
                            f"— val {_score_label}: {val_score:.4f} | "
                            f"val F1: {val_metrics['f1_best']:.4f}@thr={best_val_thr:.3f} | "
                            f"train F1: {best_train_f1:.4f}"
                        )
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[*] Early stopping at epoch {epoch + 1}.")
                        break
            else:
                if avg_loss < best_loss:
                    best_loss, epochs_no_improve = avg_loss, 0
                    best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    print(f"[*] New best model at epoch {epoch + 1} — loss: {best_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[*] Early stopping at epoch {epoch + 1}.")
                        break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            if val_dataset is not None:
                print(f"[*] Restored best model weights " f"(val F1: {best_val_f1:.4f}@thr={best_val_thr:.3f} | " f"train F1: {best_train_f1:.4f} | val AUPRC: {best_val_auprc:.4f}).")
            else:
                print(f"[*] Restored best model weights (loss: {best_loss:.4f}).")

        print("Training complete.")
        return {
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_val_f1": best_val_f1,
            "best_train_f1": best_train_f1,
            "best_val_thr": best_val_thr,
            "best_val_auprc": best_val_auprc,
            "epochs_ran": last_epoch,
        }

    # ------------------------------------------------------------------
    # Platt scaling (probability calibration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_platt_scaling(self, dataset):
        """Fit a Platt scaling calibration layer on the dataset" """
        device = next(self.parameters()).device
        self.eval()
        all_logits, all_targets = [], []
        for data in dataset:
            data = data.to(device)
            logits = self.forward(
                data.x,
                data.edge_index,
                data.edge_attr,
                edge_evidence=getattr(data, "edge_evidence", None),
            )
            all_logits.append(logits.cpu())
            all_targets.append(data.y.cpu())
        logits = torch.cat(all_logits).squeeze()
        targets = torch.cat(all_targets).squeeze().float()

        A = torch.tensor(1.0, requires_grad=True)
        B = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.LBFGS([A, B], lr=0.1, max_iter=100)

        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            cal_logits = A * logits + B
            loss = criterion(cal_logits, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        self._platt_A = float(A.item())
        self._platt_B = float(B.item())
        print(f"[*] Platt scaling fitted: A={self._platt_A:.4f}, B={self._platt_B:.4f}")
        return {"A": self._platt_A, "B": self._platt_B}

    def get_platt_params(self):
        """Return Platt scaling parameters if fitted, else None."""
        if hasattr(self, "_platt_A") and hasattr(self, "_platt_B"):
            return {"A": self._platt_A, "B": self._platt_B}
        return None

    def calibrate_logits(self, logits, platt_params=None):
        """Apply Platt scaling to raw logits"""
        params = platt_params or self.get_platt_params()
        if params is None:
            return logits
        return params["A"] * logits + params["B"]

    @torch.no_grad()
    def _log_metrics(self, logits, targets, loss, epoch, total_epochs, optimizer):
        probs = torch.sigmoid(logits)
        labels = targets.float()
        f1_05, p_05, r_05 = _f1_at_threshold(probs, labels, 0.5)
        f1_best, best_thr, p_b, r_b, ap = _f1_best_threshold(probs, labels)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d}/{total_epochs} | Loss: {loss:.4f} | "
            f"F1@0.5: {f1_05:.3f} (P={p_05:.2f} R={r_05:.2f}) | "
            f"F1@best: {f1_best:.3f}@thr={best_thr:.3f} (P={p_b:.2f} R={r_b:.2f}) | "
            f"AUPRC: {ap:.3f} | LR: {lr:.6f}"
        )


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------
def _f1_at_threshold(probs, targets, thr):
    """Calculate F1, precision and recall at a fixed threshold"""
    preds = (probs >= thr).float()
    tp = float((preds * targets).sum().item())
    fp = float((preds * (1.0 - targets)).sum().item())
    fn = float(((1.0 - preds) * targets).sum().item())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def _f1_best_threshold(probs, targets):
    """Find the threshold that maximises F1 and calculate AUPRC"""
    if targets.numel() == 0 or targets.sum().item() == 0:
        return 0.0, 0.5, 0.0, 0.0, 0.0

    sorted_idx = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_idx]
    sorted_targets = targets[sorted_idx]

    tp_cum = torch.cumsum(sorted_targets, dim=0)
    fp_cum = torch.cumsum(1.0 - sorted_targets, dim=0)
    total_pos = float(sorted_targets.sum().item())

    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / (total_pos + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(torch.argmax(f1).item())
    best_thr = float(sorted_probs[best_idx].item())
    f1_best = float(f1[best_idx].item())

    recall_padded = torch.cat([torch.zeros(1), recall])
    precision_padded = torch.cat([precision[:1], precision])
    auprc = float(torch.trapz(precision_padded, recall_padded).item())

    return f1_best, best_thr, float(precision[best_idx].item()), float(recall[best_idx].item()), auprc


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path, map_location=None):
    """Load a checkpoint file and ensure it has the expected dict structure"""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        checkpoint = {"state_dict": checkpoint}
    return checkpoint


def build_model_from_checkpoint(checkpoint_path, fallback_config=None, map_location=None):
    """Instantiate a FusionPredictor from a saved checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    config = checkpoint.get("config")

    if config is None:
        if fallback_config is None:
            raise ValueError("Checkpoint has no embedded config and no fallback_config was provided.")
        config = fallback_config

    if "edge_embed_dim" not in config:
        raise ValueError(
            "Checkpoint config is missing 'edge_embed_dim'. This checkpoint was saved with an "
            "older architecture that passed raw edge features directly to the GAT layers. "
            "Please retrain the model with the current architecture."
        )

    model = FusionPredictor(**config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint, config


def create_checkpoint(model, config, optimizer_state_dict=None, extra_metadata=None):
    """Build a checkpoint dict ready to be saved with torch.save"""
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config,
    }
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict
    if extra_metadata is not None:
        checkpoint["metadata"] = extra_metadata
    return checkpoint
