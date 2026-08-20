import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


COUNTRY_TO_INDEX = {
    "Belarus": 0,
    "Finland": 1,
    "France": 2,
    "Germany": 3,
    "Iceland": 4,
    "Italy": 5,
    "Norway": 6,
    "Poland": 7,
    "Spain": 8,
    "Sweden": 9,
    "Turkey": 10,
    "United_Kingdom": 11,
}


def build_cluster_country_matrix(
    train_csv,
    num_clusters=64,
    num_countries=12,
):
    df = pd.read_csv(train_csv)

    matrix = np.zeros(
        (
            num_clusters,
            num_countries,
        ),
        dtype=np.float32,
    )

    for cluster_id in range(num_clusters):
        rows = df[
            df["geo_cluster"] == cluster_id
        ]

        if len(rows) == 0:
            matrix[cluster_id] = (
                1.0 / num_countries
            )
            continue

        counts = (
            rows["country"]
            .value_counts()
        )

        for country_name, count in counts.items():
            country_index = COUNTRY_TO_INDEX[
                country_name
            ]

            matrix[
                cluster_id,
                country_index,
            ] = count

        row_sum = matrix[
            cluster_id
        ].sum()

        if row_sum > 0:
            matrix[
                cluster_id
            ] /= row_sum

    return matrix


class CountryConsistencyLoss(nn.Module):
    def __init__(
        self,
        cluster_country_matrix,
        eps=1e-8,
    ):
        super().__init__()

        matrix = torch.tensor(
            cluster_country_matrix,
            dtype=torch.float32,
        )

        self.register_buffer(
            "cluster_country_matrix",
            matrix,
        )

        self.eps = eps

    def forward(
        self,
        cluster_logits,
        country_logits,
    ):
        cluster_probs = F.softmax(
            cluster_logits,
            dim=1,
        )

        country_probs = F.softmax(
            country_logits,
            dim=1,
        )

        implied_country_probs = (
            cluster_probs
            @ self.cluster_country_matrix
        )

        implied_country_probs = (
            implied_country_probs
            .clamp_min(self.eps)
        )

        country_probs = (
            country_probs
            .clamp_min(self.eps)
        )

        # Symmetric KL divergence.
        #
        # This avoids treating either the cluster head
        # or the country head as unquestionably correct.
        kl_cluster_to_country = F.kl_div(
            country_probs.log(),
            implied_country_probs,
            reduction="batchmean",
        )

        kl_country_to_cluster = F.kl_div(
            implied_country_probs.log(),
            country_probs,
            reduction="batchmean",
        )

        consistency_loss = 0.5 * (
            kl_cluster_to_country
            + kl_country_to_cluster
        )

        return consistency_loss