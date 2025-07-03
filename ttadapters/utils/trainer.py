import math
import sys
import torch

model = RTDetr50ForObjectDetection.from_pretrained(num_labels=len(dataset.train.categories))

hungarian_weight_dict = {cost_class: 2, cost_bbox: 5, cost_giou: 2}
matcher = HungarianMatcher(weight_dict=hungarian_weight_dict)

loss_weight_dict = {loss_vfl: 1, loss_bbox: 5, loss_giou: 2,}
criterion = SetCriterion(matcher=matcher, weight_dict=loss_weight_dict, losses=['vfl', 'boxes'], num_classes=6)

data_loader = DatasetAdapter(dataset.train)

LEARNING_RATE = 1e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-2
)

device = ???

EPOCHS = 20

for i in range(EPOCHS):
    train_one_epoch_simple(model=model,
                           criterion=criterion,
                           data_loader=data_loader,
                           optimizer=optim,
                           device=device,
                           epoch=EPOCHS
                           )

def train_one_epoch_simple(model: torch.nn.Module,
                           criterion: torch.nn.Module,
                           data_loader: Iterable,
                           optimizer: torch.optim.Optimizer,
                           device: torch.device,
                           epoch: int,
                           max_norm: float = 0,
                           print_freq: int = 10):
    model.train()
    criterion.train()

    running_loss = 0.0
    num_batches = 0

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward + loss
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k]
                   for k in loss_dict.keys() if k in weight_dict)

        # aggregate for reporting
        loss_value = loss.item()
        running_loss += loss_value
        num_batches += 1

        # backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # 간단한 로그 출력
        if i % print_freq == 0:
            avg_loss = running_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch}][{i}/{len(data_loader)}]  "
                  f"avg_loss: {avg_loss:.4f}  lr: {lr:.6f}")
            # # 리셋(선택 사항)
            # running_loss = 0.0
            # num_batches = 0

    # 에폭이 끝난 뒤 평균 손실 계산
    # (print_freq에 맞춰 부분적으로 리셋했다면, 전체 평균을 원하면 별도로 기록 필요)
    return
