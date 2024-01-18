import torch

def train_one_epoch(dataloader, model, optimizer, scheduler, criterion, device, epoch, config):
    total_loss = 0.
    for batch in dataloader:
        sources, targets = batch

        targets = targets.to(device)
        outputs = model(sources)
    
        loss = criterion(outputs[0], targets[:, 0]) + criterion(outputs[1], targets[:, 1])
    
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    return total_loss

@torch.no_grad()
def evaluate(model, dataloader, device, epoch, config):
    model.eval()

    angular_error1, angular_error2, total = 0., 0., 0.
    for batch in dataloader:
        sources, targets = batch
        targets = targets.to(device)
        outputs = model(sources)

        angular_error1 += torch.abs(outputs[0].argmax(dim=1) - targets[:, 0]).sum() * 5
        angular_error2 += torch.abs(outputs[1].argmax(dim=1) - targets[:, 1]).sum() * 5
        # acc1 += (outputs[0].argmax(dim=1)  targets[:, 0]).sum().item()
        # acc2 += (outputs[1].argmax(dim=1) == targets[:, 1]).sum().item()
        total += len(targets)

    return angular_error1 / total, angular_error2 / total
    # acc1 = acc1 / total
    # acc2 = acc2 / total
    # return acc1, acc2