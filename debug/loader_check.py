def check_loader(train_loader):
    print("Checking DataLoader batches...")
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i}: x shape={x.shape}, y shape={y.shape}")
        if i == 2:
            break
