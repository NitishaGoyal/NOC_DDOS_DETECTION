# V5 P2 E1-P2 Exact Source Interface Resolution

## Resolution summary

- Loader: `V5P2PairAlignedPrimary58Dataset` from `src/data/v5_p2_pair_aligned_primary58_dataset.py`
- Sampler: `PairBlockBatchSampler` from `scripts/v5/p2/train_v5_p2_b2_single_seed.py`
- Task-D model: `P2TaskDGraphConvCount4` from `src/models/v5_p2_task_d_full_multitask_count4.py`
- B3 model: `P2B3Conv1DOnlyCount4` from `src/models/v5_p2_b3_conv1d_only_count4.py`
- Exporter ready: **false**
- Validation tensors opened: **false**
- Test directory enumerated: **false**

## Exact loader interface

```json
{
  "class": "V5P2PairAlignedPrimary58Dataset",
  "constructor_signature": "self, root: str | Path, split: str, pair_manifest: str | Path, *, window: int=WINDOW, stride: int=STRIDE",
  "getitem_return_keys": [
    "physical_port_mask",
    "role_mask",
    "x",
    "y_attack",
    "y_attack_path",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim"
  ],
  "getitem_signature": "self, index: int",
  "metadata_keys_available": [],
  "missing_required_label_keys": [
    "y_graph",
    "y_path"
  ],
  "path": "/home/zira/research/projects/GNN-2d/src/data/v5_p2_pair_aligned_primary58_dataset.py"
}
```

### Loader `__init__`

```python
def __init__(
        self,
        root: str | Path,
        split: str,
        pair_manifest: str | Path,
        *,
        window: int = WINDOW,
        stride: int = STRIDE,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.pair_manifest = Path(pair_manifest).expanduser().resolve()
        self.window = int(window)
        self.stride = int(stride)

        if self.split not in AUTHORIZED_SPLITS:
            raise ValueError(
                "This audited loader authorizes only train and validation. "
                "The P2 test split remains locked."
            )
        if self.window != WINDOW:
            raise ValueError(
                f"window must remain frozen at {WINDOW}, got {self.window}"
            )
        if self.stride != STRIDE:
            raise ValueError(
                f"stride must remain frozen at {STRIDE}, got {self.stride}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        if not self.pair_manifest.is_file():
            raise FileNotFoundError(
                f"pair manifest missing: {self.pair_manifest}"
            )

        split_dir = self.root / "runs" / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"authorized split directory missing: {split_dir}"
            )

        topology = torch.load(
            self.root / "topology.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
        self._pair_count = 0

        with self.pair_manifest.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        seen_pairs: set[str] = set()
        for row in rows:
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
            self._pair_count += 1

            common_length = _as_int(row, "common_length")
            window_count = _as_int(
                row,
                "window_count_per_member",
            )
            manifest_window = _as_int(row, "window")
            manifest_stride = _as_int(row, "stride")

            if manifest_window != self.window:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window="
                    f"{manifest_window}, expected {self.window}"
                )
            if manifest_stride != self.stride:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest stride="
                    f"{manifest_stride}, expected {self.stride}"
                )

            expected_window_count = (
                0
                if common_length < self.window
                else 1
                + (common_length - self.window) // self.stride
            )
            if window_count != expected_window_count:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window count="
                    f"{window_count}, expected {expected_window_count}"
                )

            attack_path = split_dir / f"{pair_key}_ATTACK.pt"
            control_path = split_dir / f"{pair_key}_CONTROL.pt"
            if not attack_path.is_file():
                raise FileNotFoundError(
                    f"missing ATTACK tensor: {attack_path}"
                )
            if not control_path.is_file():
                raise FileNotFoundError(
                    f"missing CONTROL tensor: {control_path}"
                )

            for window_index in range(window_count):
                start = window_index * self.stride
                target = start + self.window - 1

                if target >= common_length:
                    raise ValueError(
                        f"{self.split}/{pair_key}: target {target} "
                        f"outside common length {common_length}"
                    )

                # Pair-major, start-major, ATTACK then CONTROL.
                self._index.append(
                    IndexEntry(
                        file_path=attack_path,
                        pair_key=pair_key,
                        mode="attack",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )
                self._index.append(
                    IndexEntry(
                        file_path=control_path,
                        pair_key=pair_key,
                        mode="control",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )

        if self._pair_count == 0:
            raise ValueError(
                f"pair manifest contains no rows for split {self.split!r}"
            )
        if not self._index:
            raise ValueError(
                f"no aligned windows produced for split {self.split!r}"
            )
        if len(self._index) % 2 != 0:
            raise RuntimeError(
                "pair-aligned dataset length must be even"
            )
```

### Loader `__len__`

```python
def __len__(self) -> int:
        return len(self._index)
```

### Loader `__getitem__`

```python
def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._index[index]
        run = self._load_run(entry.file_path)

        x_stored = run.get("x")
        if (
            not isinstance(x_stored, torch.Tensor)
            or x_stored.dtype != torch.float32
            or x_stored.ndim != 3
            or tuple(x_stored.shape[1:]) != (16, 81)
        ):
            raise ValueError(
                f"invalid stored x tensor in {entry.file_path}"
            )
        if x_stored.shape[0] < entry.common_length:
            raise ValueError(
                f"{entry.file_path} length={x_stored.shape[0]} "
                f"is below common length={entry.common_length}"
            )

        stop = entry.start + self.window
        x = (
            x_stored[
                entry.start:stop,
                :,
                PRIMARY58_INDICES,
            ]
            .permute(1, 2, 0)
            .contiguous()
        )

        if tuple(x.shape) != (16, 58, 32):
            raise RuntimeError(
                f"constructed x shape={tuple(x.shape)}, "
                "expected [16,58,32]"
            )

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
                run["y_attacker_count"][entry.target].clone()
            ),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": (
                run["y_attack_path"][entry.target].clone()
            ),
            "role_mask": run["role_mask"][entry.target].clone(),
        }

        if set(item) != MODEL_ITEM_KEYS:
            raise RuntimeError("loader item-key contract changed")
        return item
```

## Exact sampler interface

```json
{
  "class": "PairBlockBatchSampler",
  "constructor_signature": "self, dataset, *, block_batch_size: int, shuffle: bool, seed: int",
  "iter_signature": "self",
  "path": "/home/zira/research/projects/GNN-2d/scripts/v5/p2/train_v5_p2_b2_single_seed.py"
}
```

### Sampler `__init__`

```python
def __init__(
        self,
        dataset,
        *,
        block_batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.block_batch_size = int(block_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if self.block_batch_size <= 0:
            raise ValueError("block_batch_size must be positive")
        if len(dataset) % 2 != 0:
            raise ValueError("dataset item count must be even")

        groups: OrderedDict[str, list[int]] = OrderedDict()
        for base in range(0, len(dataset._index), 2):
            attack = dataset._index[base]
            control = dataset._index[base + 1]

            if (
                attack.mode != "attack"
                or control.mode != "control"
                or attack.pair_key != control.pair_key
                or attack.start != control.start
                or attack.target != control.target
            ):
                raise ValueError(
                    f"pair-block contract failed at base {base}"
                )

            groups.setdefault(
                attack.pair_key,
                [],
            ).append(base)

        self.groups = groups
        self.block_count = sum(
            len(value)
            for value in groups.values()
        )

        if self.block_count * 2 != len(dataset):
            raise RuntimeError("pair-block count does not cover dataset")
```

### Sampler `__iter__`

```python
def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(
            self.seed + self.epoch * 1_000_003
        )

        pair_keys = list(self.groups)
        if self.shuffle:
            rng.shuffle(pair_keys)

        ordered_blocks: list[int] = []
        for pair_key in pair_keys:
            blocks = list(self.groups[pair_key])
            if self.shuffle:
                rng.shuffle(blocks)
            ordered_blocks.extend(blocks)

        for start in range(
            0,
            len(ordered_blocks),
            self.block_batch_size,
        ):
            selected_blocks = ordered_blocks[
                start:start + self.block_batch_size
            ]
            item_indices: list[int] = []
            for base in selected_blocks:
                item_indices.extend((base, base + 1))
            yield item_indices
```

## Exact Task-D model interface

```json
{
  "class": "P2TaskDGraphConvCount4",
  "constructor_signature": "self, reference_b3: nn.Module, base_edge_index: torch.Tensor",
  "forward_return_keys": [
    "attack_logits",
    "count_logits",
    "path_logits",
    "source_logits",
    "transit_logits",
    "victim_logits"
  ],
  "forward_signature": "self, x: torch.Tensor, physical_port_mask: torch.Tensor",
  "missing_required_output_keys": [],
  "path": "/home/zira/research/projects/GNN-2d/src/models/v5_p2_task_d_full_multitask_count4.py"
}
```

### Task-D model `__init__`

```python
def __init__(self, reference_b3: nn.Module, base_edge_index: torch.Tensor) -> None:
        super().__init__()
        if GraphConv is None:
            raise RuntimeError(f"torch_geometric GraphConv unavailable: {_PYG_IMPORT_ERROR}")

        # Exact common B3 components.
        self.input_projection = copy.deepcopy(reference_b3.input_projection)
        self.temporal_blocks = copy.deepcopy(reference_b3.temporal_blocks)
        self.node_projection = copy.deepcopy(reference_b3.node_projection)
        self.source_head = copy.deepcopy(reference_b3.source_head)
        self.transit_head = copy.deepcopy(reference_b3.transit_head)
        self.victim_head = copy.deepcopy(reference_b3.victim_head)
        self.path_head = copy.deepcopy(reference_b3.path_head)
        self.graph_projection = copy.deepcopy(reference_b3.graph_projection)
        self.attack_head = copy.deepcopy(reference_b3.attack_head)
        self.count_head = copy.deepcopy(reference_b3.count_head)

        self.register_buffer(
            "base_edge_index",
            base_edge_index.detach().cpu().long().contiguous(),
        )
        self.graph1 = GraphConv(64, 64, aggr="add", bias=True)
        self.graph2 = GraphConv(64, 64, aggr="add", bias=True)
        self.activation = nn.ReLU()

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                f"Task-D GraphConv parameter count={parameter_count}, "
                f"expected {self.expected_parameter_count}"
            )
```

### Task-D model `forward`

```python
def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
            self.base_edge_index,
            batch_size,
            num_nodes,
            flat.device,
        )
        flat = self.activation(self.graph1(flat, edges))
        flat = self.activation(self.graph2(flat, edges))
        node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs
```

## Frozen validation construction

### Dataset And Loader

Match line 201 (context starts 193):

```python
    b3_mod = import_module(paths["b3_model"], f"taskd_b3_{candidate}_{seed}")
    td_mod = import_module(paths["task_d_model"], f"taskd_model_{candidate}_{seed}")

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
```

Match line 203 (context starts 195):

```python

    b2.EXPECTED_SEEDS = SEEDS
    b2.EXPECTED_TRAIN_ITEMS = EXPECTED_TRAIN_ITEMS
    b2.EXPECTED_VALIDATION_ITEMS = EXPECTED_VALIDATION_ITEMS
    b2.set_seed(seed)

    DatasetClass = loader_mod.V5P2PairAlignedPrimary58Dataset
    train_dataset = DatasetClass(root=root, split="train", pair_manifest=paths["pair_manifest"])
    validation_dataset = DatasetClass(root=root, split="validation", pair_manifest=paths["pair_manifest"])
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(f"validation length={len(validation_dataset)}")

    train_summary = b0_report["label_summaries"]["train"]
    graph_positive = int(train_summary["graph_positive"])
    graph_negative = int(train_summary["graph_negative"])
```

Match line 239 (context starts 231):

```python
        }
    count_distribution = b2.parse_distribution(train_summary["active_count_distribution"])
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
```

Match line 241 (context starts 233):

```python
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
```

### Model Construction

Match line 243 (context starts 235):

```python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
```

Match line 244 (context starts 236):

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")
```

Match line 248 (context starts 240):

```python
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMS[candidate]:
        raise RuntimeError(f"parameter count={parameter_count}, expected {EXPECTED_PARAMS[candidate]}")

    full_initial_state_sha256 = b2.sha256_state_dict(model.state_dict())
    if candidate == "conv1d":
        common_initial_state = OrderedDict((k, v) for k, v in model.state_dict().items())
```

### Checkpoint Load

Match line 396 (context starts 388):

```python
        )
        if epoch >= 15 and patience_counter >= 12:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
        "stage": STAGE,
```

Match line 397 (context starts 389):

```python
        if epoch >= 15 and patience_counter >= 12:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
```

### Validation Call

Match line 241 (context starts 233):

```python
    if sorted(count_distribution) != COUNT_CLASS_VALUES:
        raise RuntimeError("count distribution is not 1..4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = b2.PairBlockBatchSampler(train_dataset, block_batch_size=128, shuffle=True, seed=seed)
    validation_sampler = b2.PairBlockBatchSampler(validation_dataset, block_batch_size=128, shuffle=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=0, pin_memory=pin_memory)

    reference_b3 = b3_mod.P2B3Conv1DOnlyCount4()
    edge = torch.from_numpy(np.load(paths["edge"], allow_pickle=False)).long()
    if candidate == "conv1d":
        model = reference_b3
    else:
        model = td_mod.P2TaskDGraphConvCount4(reference_b3, edge)
    model = model.to(device)
```

Match line 311 (context starts 303):

```python
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = b2.train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
            best_epoch = epoch
```

Match line 312 (context starts 304):

```python
        train_sampler.set_epoch(epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = b2.train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
            gradient_clip=1.0,
        )
        validation_metrics = b2.validate(
            model=model, loader=validation_loader, device=device,
            graph_pos_weight=graph_pos_weight, role_pos_weights=role_pos_weights,
        )
        rank = b2.checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
            best_epoch = epoch
            best_validation_metrics = validation_metrics
```

Match line 398 (context starts 390):

```python
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached 12")
            break

    if best_epoch is None or best_validation_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    first_validation_batch = next(iter(validation_loader))
    latency_us = measure_latency(model, first_validation_batch, device)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    operation_proxy = td_mod.operation_count_proxy(candidate)

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "candidate": candidate,
```

Match line 422 (context starts 414):

```python
            "task_d_model_sha256": sha256_file(paths["task_d_model"]),
            "loader_sha256": sha256_file(paths["loader"]),
            "b1_protocol_sha256": protocol["protocol_sha256"],
            "promotion_report_sha256": sha256_file(paths["promotion_report"]),
        },
        "data": {
            "train_items": len(train_dataset), "validation_items": len(validation_dataset),
            "pair_block_batch_size": 128, "item_batch_size": 256,
            "train_batches": len(train_loader), "validation_batches": len(validation_loader),
            "pair_keys_returned_to_model": False,
        },
        "label_weights": {
            "graph_positive_weight": graph_pos_weight_value,
            "role_positive_weights": role_weight_manifest,
            "count_distribution": count_distribution,
            "count_class_weights_applied": False,
        },
```

## Stable identity rule

```json
{
  "canonical_string_rule": "join present components as key=value using ASCII unit separator \\x1f in the component order above",
  "components_in_frozen_loader_order": [],
  "fallback_rule": "If the loader does not expose enough components to guarantee uniqueness, E1 must add immutable validation_item_index in emitted loader order and must not claim semantic identity beyond that index.",
  "status": "PARTIALLY_RESOLVED"
}
```

## Current state

- E1 exporter implemented: **false**
- Validation predictions exported: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
