from Model.py import *

def main_train(config, continue_= None):
    # Check if files exist
    for file_key in ['smomp_file', 'accurate_file', 'user_positions_file', 'rss_image_path']:
        if not os.path.exists(config[file_key]):
            print(f"Error: {config[file_key]} not found!")
    
    print(f"Using device: {config['device']}")
    
    # Initialize RSS processor
    rss_processor = RSSMapProcessor(
        image_path=config['rss_image_path'],
        bs_pixel_coords=config['bs_pixel_coords'],
        bs_real_coords=config['bs_real_coords'],
        image_width_meters=config['image_width_meters']
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config['smomp_file'], config['accurate_file'], 
        config['user_positions_file'], rss_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False)
    
    # Initialize model
    #model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=config['epochs'], 
        lr=config['learning_rate'],
        device=config['device'], 
        model_name_val = config['name_val'],
        model_name_train = config['name_train'], continue_ = continue_
        )
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set on best val...")
    model.load_state_dict(torch.load(config['name_val']))
    test_nmse = evaluate_test_set(model, test_loader, device=config['device'])
    
    print(f"\nFinal Test NMSE: {test_nmse:.6f}")
    print(f"Test NMSE in dB: {10 * np.log10(test_nmse):.2f} dB")

    print("\nEvaluating on test set on best train...")
    checkpoint = torch.load(config['name_train'])
    model.load_state_dict(checkpoint['model_state_dict'])
    test_nmse = evaluate_test_set(model, test_loader, device=config['device'])
    
    print(f"\nFinal Test NMSE: {test_nmse:.6f}")
    print(f"Test NMSE in dB: {10 * np.log10(test_nmse):.2f} dB")
    
    # Plot training curves
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(val_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Loss')
    # plt.title('Validation Loss')
    
    # plt.tight_layout()
    # plt.savefig('training_curves.png')
    # plt.show()

    return model, val_loader, test_loader

if __name__ == "__main__":
    # Configuration
    config = {
        'smomp_file': 'initial_estimate_ls_snr0.npy',
        'accurate_file': '3D_channel_15GHz_2x2_Pt50.npy',
        'user_positions_file': 'ue_positions_noisy.txt',
        'rss_image_path': '50.jpg',
        'bs_pixel_coords': (287, 293),
        'bs_real_coords': (71.06, 246.29),
        'image_width_meters': 527.5,
        'batch_size': 32,
        'epochs': 500,
        'learning_rate': 1e-3,
        'device': 'cuda',
        'name_val':'simple_ls_0_val.pth',
        'name_train':'simple_ls_0_train.pth'
    }
    model = main_train(config)
