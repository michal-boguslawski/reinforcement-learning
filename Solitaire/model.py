import torch
import torch.nn as nn


class SolitaireModel(nn.Module):
    def __init__(self, embedding_dims, dense_units, num_layers=2, nhead=2, device='cpu'):
        super(SolitaireModel, self).__init__()
        #self.param = nn.Parameter(torch.randn(1, 1, embedding_dims[0]+embedding_dims[1], device=device))
        self.pos_enc = nn.Parameter(torch.randn(1, 145, dense_units, device=device))
        self.color_embedding = nn.Embedding(6, embedding_dims[0], padding_idx=0, device=device)
        self.figure_embedding = nn.Embedding(15, embedding_dims[1], padding_idx=0, device=device)
        self.dense0 = nn.Sequential(
            nn.Linear(embedding_dims[0]+embedding_dims[1], dense_units, device=device),
            nn.LayerNorm(dense_units, device=device),
            nn.GELU()
        )
        self.transformer_enc_layer = nn.TransformerEncoderLayer(
            dense_units,
            nhead=nhead,
            dim_feedforward=dense_units*4,
            device=device
        )
        self.transformer_layer = nn.TransformerEncoder(
            self.transformer_enc_layer,
            num_layers
        )
        self.output_tab_prepare = nn.Sequential(nn.MaxPool2d((20, 1)),
                                                nn.Flatten(2, 3),)
        self.output_tab_dense = nn.Linear(dense_units, 9, device=device)

        self.output_found_prepare = nn.Sequential(nn.MaxPool1d(4),
                                                  )
        self.output_found_dense = nn.Linear(dense_units, 9, device=device)

        self.output_waste_dense = nn.Linear(dense_units, 9, device=device)

    def forward(self, inputs):
        temp_bs = len(inputs)
        color_layer = self.color_embedding(inputs[..., 0])
        figure_layer = self.figure_embedding(inputs[..., 1])
        x = torch.cat([color_layer, figure_layer], dim=-1)
        x = self.dense0(x)
        x = torch.add(x, self.pos_enc.repeat(temp_bs, 1, 1))
        # x = torch.cat([self.param.repeat((temp_bs, 1, 1)), x], dim=1)
        x = self.transformer_layer(x)
        # x = x[:, 0]

        tableau = x[:, :140]
        tableau = tableau.view(temp_bs, 20, 7, -1)
        tableau = tableau.permute(0, 3, 1, 2)
        tableau = self.output_tab_prepare(tableau)
        tableau = tableau.permute(0, 2, 1)
        tableau = self.output_tab_dense(tableau)

        foundation = x[:, 140:144]
        foundation = foundation.permute(0, 2, 1)
        foundation = self.output_found_prepare(foundation)
        foundation = foundation.permute(0, 2, 1)
        foundation = self.output_found_dense(foundation)

        waste = x[:, 144:145]
        waste = self.output_waste_dense(waste)

        output = torch.cat([tableau, foundation, waste], dim=1)
        return output.view(temp_bs, 9 * 9)


class SolitaireCNNAttModel(nn.Module):
    def __init__(self, embedding_dims, dense_units, num_layers=2, nhead=2, device='cpu'):
        super(SolitaireCNNAttModel, self).__init__()
        #self.param = nn.Parameter(torch.randn(1, 1, embedding_dims[0]+embedding_dims[1], device=device))
        self.pos_enc = nn.Parameter(torch.randn(1, 9, dense_units, device=device))
        self.color_embedding = nn.Embedding(6, embedding_dims[0], padding_idx=0, device=device)
        self.figure_embedding = nn.Embedding(15, embedding_dims[1], padding_idx=0, device=device)

        self.conv_tableau = nn.Sequential(
            nn.Conv2d(embedding_dims[0] + embedding_dims[1], dense_units, (3, 1), (2, 1), device=device),
            nn.InstanceNorm2d(dense_units),
            nn.GELU(),
            nn.Conv2d(dense_units, dense_units, (3, 1), (2, 1), device=device),
            nn.InstanceNorm2d(dense_units),
            nn.GELU(),
            nn.Conv2d(dense_units, dense_units, (4, 1), device=device),
            # nn.InstanceNorm2d(dense_units),
            nn.GELU(),
            nn.Flatten(2, 3)
        )

        self.conv_foundation = nn.Sequential(
            nn.Conv1d(embedding_dims[0] + embedding_dims[1], dense_units, 4, device=device),
            # nn.LayerNorm(dense_units),
            nn.GELU()
        )

        self.waste_dense = nn.Sequential(
            nn.Linear(embedding_dims[0] + embedding_dims[1], dense_units, device=device),
            nn.LayerNorm(dense_units, device=device),
            nn.GELU()
        )

        self.transformer_enc_layer = nn.TransformerEncoderLayer(
            dense_units,
            nhead=nhead,
            dim_feedforward=dense_units*4,
            device=device
        )
        self.transformer_layer = nn.TransformerEncoder(
            self.transformer_enc_layer,
            num_layers
        )
        self.output_tab_prepare = nn.Sequential(nn.MaxPool2d((20, 1)),
                                                nn.Flatten(2, 3),)
        self.output_tab_dense = nn.Linear(dense_units, 9, device=device)

        self.output_found_prepare = nn.Sequential(nn.MaxPool1d(4),
                                                  )
        self.output_found_dense = nn.Linear(dense_units, 9, device=device)

        self.output_waste_dense = nn.Linear(dense_units, 9, device=device)

    def forward(self, inputs):
        temp_bs = len(inputs)
        color_layer = self.color_embedding(inputs[..., 0])
        figure_layer = self.figure_embedding(inputs[..., 1])
        x = torch.cat([color_layer, figure_layer], dim=-1)

        tableau = x[:, :140]
        tableau = tableau.view(temp_bs, 20, 7, -1)
        tableau = tableau.permute(0, 3, 1, 2)
        tableau = self.conv_tableau(tableau)
        # tableau = tableau[:, :, 0]
        tableau = tableau.permute(0, 2, 1)

        foundation = x[:, 140:144]
        foundation = foundation.permute(0, 2, 1)
        foundation = self.conv_foundation(foundation)
        foundation = foundation.permute(0, 2, 1)

        waste = x[:, 144:145]
        waste = self.waste_dense(waste)

        x = torch.cat([tableau, foundation, waste], 1)
        x = torch.add(x, self.pos_enc.repeat(temp_bs, 1, 1))
        # x = torch.cat([self.param.repeat((temp_bs, 1, 1)), x], dim=1)
        x = self.transformer_layer(x)
        # x = x[:, 0]

        tableau = x[:, :7]
        tableau = self.output_tab_dense(tableau)

        foundation = x[:, 7:8]
        foundation = self.output_found_dense(foundation)

        waste = x[:, 8:9]
        waste = self.output_waste_dense(waste)

        output = torch.cat([tableau, foundation, waste], dim=1)
        return output.view(temp_bs, 9 * 9)
