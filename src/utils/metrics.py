import torch

LAT_MEAN = 52.65384082447373
LAT_STD = 9.87951259488397

LNG_MEAN = 11.316609908569
LNG_STD = 15.245180654282903


def denormalize_coordinates(coords):
    lat = coords[:, 0] * LAT_STD + LAT_MEAN
    lng = coords[:, 1] * LNG_STD + LNG_MEAN

    return torch.stack([lat, lng], dim=1)



def haversine_distance_km(preds, targets):
    """
    Compute Haversine distance in kilometers.

    preds:   tensor of shape [batch_size, 2]
             columns = [latitude, longitude]

    targets: tensor of shape [batch_size, 2]
             columns = [latitude, longitude]
    """

    radius_earth_km = 6371.0088

    pred_lat = torch.deg2rad(preds[:, 0])
    pred_lng = torch.deg2rad(preds[:, 1])

    true_lat = torch.deg2rad(targets[:, 0])
    true_lng = torch.deg2rad(targets[:, 1])

    delta_lat = true_lat - pred_lat
    delta_lng = true_lng - pred_lng

    a = (
        torch.sin(delta_lat / 2) ** 2
        + torch.cos(pred_lat)
        * torch.cos(true_lat)
        * torch.sin(delta_lng / 2) ** 2
    )

    a = torch.clamp(a, 0.0, 1.0)

    distance = 2 * radius_earth_km * torch.asin(torch.sqrt(a))

    return distance


def median_haversine_km(preds, targets):
    distances = haversine_distance_km(preds, targets)
    return torch.median(distances)
