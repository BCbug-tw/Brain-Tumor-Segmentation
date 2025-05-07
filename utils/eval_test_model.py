import torch
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from utils.utils_func import dice_score, precision_recall, natural_key

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_dice += dice_score(outputs, masks).item()
            precision, recall = precision_recall(outputs, masks)
            total_precision += precision.item()
            total_recall += recall.item()

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    return avg_loss, avg_dice, avg_precision, avg_recall

def test_model(model, dataloader, device, output_dir="results"):
    file_name = sorted(os.listdir("data/test/images"), key=natural_key)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            for j in range(images.size(0)):
                pred_path = os.path.join(output_dir, f"{file_name[i][:-4]}_pred.jpg")
                gt_path = os.path.join(output_dir, f"{file_name[i][:-4]}_gt.jpg")
                overlay_path = os.path.join(output_dir, f"overlay/{file_name[i][:-4]}_overlay.jpg")
                
                save_image(preds[j], pred_path)
                save_image(masks[j], gt_path)

                # Overlay visualization
                pred_np = preds[j].squeeze().cpu().numpy()
                mask_np = masks[j].squeeze().cpu().numpy()
                image_np = images[j].cpu().permute(1, 2, 0).numpy()
                
                fig, ax = plt.subplots(1, 3, figsize=(12, 5))
                ax[0].imshow(image_np)
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(mask_np, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")
                
                plt.imshow(pred_np, cmap='inferno')
                plt.imshow(mask_np, cmap='gray', alpha=0.5)
                ax[2].set_title("Predition Overlay")
                ax[2].axis("off")

                plt.clim(0,1)
                plt.savefig(overlay_path)
                plt.close()

                total_dice += dice_score(outputs[j:j+1], masks[j:j+1]).item()
                precision, recall = precision_recall(outputs[j:j+1], masks[j:j+1])
                total_precision += precision.item()
                total_recall += recall.item()

    avg_dice = total_dice / len(dataloader.dataset)
    avg_precision = total_precision / len(dataloader.dataset)
    avg_recall = total_recall / len(dataloader.dataset)      
    
    print(f"Testing is done. Results is saved in ./{output_dir}\nAverage Dice Score: {avg_dice:.4f}\nAverage Precision: {avg_precision:.4f}\nAverage Recall: {avg_recall:.4f}")