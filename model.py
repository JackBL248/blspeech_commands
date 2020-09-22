import copy
import time

import torch
import torch.nn as nn
from torchvision import models

from config import Config


class Model(object):

    def __init__(self, model_type, dropout):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if model_type == "alexnet":
            self.model = models.alexnet(pretrained=False)
            self.model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=4096,
                    out_features=Config.NUM_CLASSES
            )
            self.model = self.model.to(self.device)

        if model_type == "resnet18":
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=Config.NUM_CLASSES
                )
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        if model_type == "resnet34":
            self.model = models.resnet34(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=Config.NUM_CLASSES
                )
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        if model_type == "resnet50":
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=Config.NUM_CLASSES
                )
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        if model_type == "densenet121":
            self.model = models.densenet121(pretrained=False)
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=1024,
                    out_features=Config.NUM_CLASSES
                )
            )
            self.model.classifier = classifier
            self.model = self.model.to(self.device)

        else:
            raise ValueError("Model type not recognised. \
                Please use one of the following models: alexnet, resnet18, \
                resnet34, resnet50 or densenet121")

    def train_model(
        self,
        model,
        criterion,
        optimizer,
        dataloaders,
        dataset_sizes,
        num_epochs=25,
        patience=15,
    ):
        """
        Trains a neural network with early stopping.
        -----------------------
        Args: PyTorch model, training criterion, optimizer, num epochs
        and early stopping patience

        Returns: trained model with the best weights saved
        -----------------------
        """
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 1000.00
        # Set counter for early stopping
        early_stop_counter = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        # Reset early stopping counter to zero if there is
                        # improvement in loss
                        early_stop_counter = 0

                    elif epoch_loss >= best_loss:
                        # Increment early stopping counter if there is no
                        # improvement in loss
                        early_stop_counter += 1

                # If loss has not improved for 15 epochs, stop the training
            if early_stop_counter == patience:
                break

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def test_model()