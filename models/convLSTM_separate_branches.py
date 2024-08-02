import torch
import torch.nn as nn
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob)
        )
    
    def forward(self, x):
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dropout_prob, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.dropout = nn.Dropout2d(dropout_prob)
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        combined_conv = self.dropout(combined_conv)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) #split combined conv into 4 (each has as many channels as self.hidden_dim)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout_prob):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.cell_list = nn.ModuleList([ConvLSTMCell(input_dim, hidden_dim, kernel_size, self.dropout_prob)])
        for _ in range(1, num_layers):
            self.cell_list.append(ConvLSTMCell(hidden_dim, hidden_dim, kernel_size, self.dropout_prob))

    def forward(self, input_tensor):
        b, seq_len, c, h, w = input_tensor.size()
        hidden_states = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            cur_layer_input = torch.stack(output_inner, dim=1)
        return cur_layer_input[:, -1, :, :, :]

class ConvLSTMSeparateBranches(nn.Module):
    def __init__(self, preceding_rainfall_days, forecast_rainfall_days, dropout_prob):
        super(ConvLSTMSeparateBranches, self).__init__()
        self.preceding_rainfall_days = preceding_rainfall_days
        self.forecast_rainfall_days = forecast_rainfall_days
        self.rainfall_sequence_length = preceding_rainfall_days + forecast_rainfall_days
        self.dropout_prob = dropout_prob
        self.name = "convLSTM_separate_branches"

        self.conv1 = ConvBlock(1, 16, self.dropout_prob)
        self.conv2 = ConvBlock(1, 16, self.dropout_prob)
        self.convlstm = ConvLSTM(1, 16, 3, 1, self.dropout_prob)
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            # nn.Sigmoid() # No sigmoid as BCELogitsLoss as a criterion is more stable
        )

    def forward(self, x):
        b, c, h, w = x.size()
        rainfall_sequence = x[:, :self.rainfall_sequence_length].view(b, self.rainfall_sequence_length, 1, h, w)
        topology = x[:, self.rainfall_sequence_length:self.rainfall_sequence_length+1].view(b, 1, h, w)
        soil_moisture = x[:, self.rainfall_sequence_length+1:self.rainfall_sequence_length+2].view(b, 1, h, w)
        
        rainfall_out = self.convlstm(rainfall_sequence)
        topology_out = self.conv1(topology)
        soil_moisture_out = self.conv2(soil_moisture)

        combined = torch.cat([rainfall_out, topology_out, soil_moisture_out], dim=1)
        output = self.final_conv(combined) #NCXY, where C is 1
        output = output.squeeze(1) #NXY
        return output
    