import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *multi_inputs):
        if len(multi_inputs)==2:
            logits = self.model(multi_inputs[0], multi_inputs[1])
        # if model_type in ['Multi_3DResNet', 'DAFT', 'MultiRes']:
        else:
            logits = self.model(multi_inputs[0])
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, model_type):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():

            if str(valid_loader.dataset.__class__) == "<class '__main__.MultiDataset'>":
                for input, tabular, label  in valid_loader:
                    input = input.cuda()
                    if model_type in ['Multi_3DResNet','DAFT','MultiRes','MultiLight']:
                        tabular= tabular.cuda()
                        logits = self.model(input,tabular)
                    else:
                        logits = self.model(input)
                    logits_list.append(logits)
                    label = label.type(torch.LongTensor)
                    labels_list.append(label)
            else:
                for input, label in valid_loader:
                    input = input.cuda()
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self, after_temperature_ece


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=30):   #was 15
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):

        temperature_df = pd.DataFrame(columns=(['conf_lower','conf_upper','prop_in_bin','acc_in_bin','av_conf_in_bin','ece']))
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        i=0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            temperature_df.loc[i, 'conf_lower'] = bin_lower.item()
            temperature_df.loc[i, 'conf_upper'] = bin_upper.item()
            temperature_df.loc[i, 'prop_in_bin'] = prop_in_bin.item()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                temperature_df.loc[i, 'acc_in_bin'] = accuracy_in_bin.item()
                temperature_df.loc[i, 'av_conf_in_bin'] = avg_confidence_in_bin.item()
                temperature_df.loc[i, 'ece'] = (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item()
                i+=1
        temperature_df.to_excel('temperature_data.xlsx')
        return ece
