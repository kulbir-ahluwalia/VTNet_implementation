import os
import argparse
import time
from datetime import datetime
from data.prevtdataset import PreVTDataset
from models.vtnet import PreTrainedVT
from globals import *
from torch.utils.data import DataLoader


def createOutputDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

    path = f"{path}/{datetime.now().strftime('%Y%m%d-%H%M%S%f')}"
    os.mkdir(path)
    return path


def train(model, criterion, data_loader, optimizer, epoch, print_every=1):
    model.train()
    criterion.train()

    start = time.time()
    for idx, (global_feature, local_feature, optimal_action) in enumerate(data_loader):
        optimizer.zero_grad()
        global_feature = global_feature.to(device)
        optimal_action = optimal_action.to(device)
        local_feature = {
            k: v.float().to(device) for k, v in local_feature.items() if not k in ("locations", "targets")
        }

        output = model(global_feature, local_feature)
        loss = criterion(output["action"], optimal_action)
        loss.backward()
        optimizer.step()
        if (idx + 1) % print_every == 0:
            print(f"epoch: {epoch}, iter: {idx}, loss: {loss.item()}, time: {time.time() - start} sec")
            start = time.time()


@torch.no_grad()
def evaluate(model, data_loader, epoch):
    model.eval()

    total = 0
    correct = 0
    for idx, (global_feature, local_feature, optimal_action) in enumerate(data_loader):
        global_feature = global_feature.to(device)
        optimal_action = optimal_action.to(device)
        local_feature = {
            k: v.float().to(device) for k, v in local_feature.items() if not k in ("locations", "targets")
        }

        output = model(global_feature, local_feature)
        predicted = output["action"].argmax(dim=1)
        total += predicted.size(0)
        correct += (predicted == optimal_action).sum().item()

    print(f"epoch: {epoch}, pre-training accuracy: {correct / total}")


def main(args):
    out_dir = createOutputDirectory(args.out_dir)
    model = PreTrainedVT(device, use_nn_transformer=args.use_nn_transformer).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.1)

    train_dataset = PreVTDataset(args.data_dir, split="train")
    val_dataset = PreVTDataset(args.data_dir, split="val")
    test_dataset = PreVTDataset(args.data_dir, split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)

    # Training
    print(f"Starting pre-training...")
    start_epoch = 0
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train(
            model=model,
            criterion=criterion,
            data_loader=train_dataloader,
            optimizer=optimizer,
            epoch=epoch+1,
        )
        lr_scheduler.step()

        evaluate(
            model=model,
            data_loader=val_dataloader,
            epoch=epoch+1,
        )

        checkpoints = [os.path.join(out_dir, "checkpoint.pth")]
        if (epoch + 1) % args.save_every == 0:
            checkpoints.append(os.path.join(out_dir, f"checkpoint-{epoch + 1}.pth"))

        for checkpoint in checkpoints:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }, checkpoint)

        print(f"epoch: {epoch + 1} finished, time: {time.time() - start} sec")

    # Testing
    print("Starting testing...")
    evaluate(
        model=model,
        data_loader=test_dataloader,
        epoch=-1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VTNet pretraining.")
    parser.add_argument("--data-dir", type=str, required=True, dest="data_dir", help="Data directory of pretraining data")
    parser.add_argument("--out-dir", type=str, required=True, dest="out_dir", help="Output directory")
    parser.add_argument("--lr", type=float, required=False, dest="lr", help="Learning rate", default=0.0001)
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", help="Weight decay", default=0.0001)
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", help="Batch size", default=32)
    parser.add_argument("--num-workers", type=int, required=False, dest="num_workers", help="Number of workers", default=0)
    parser.add_argument("--epochs", type=int, required=False, dest="n_epochs", help="Number of epochs", default=100)
    parser.add_argument("--do-test", action="store_true", dest="do_test", help="Perform testing")
    parser.add_argument("--save-every", type=int, required=False, dest="save_every", help="Save trained models after {save-every} epochs", default=1)
    parser.add_argument("--use-nn-transformer", action="store_true", dest="use_nn_transformer", help="Use torch.nn.Transformer")

    args = parser.parse_args()
    main(args)
