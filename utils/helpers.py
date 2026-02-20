def confidence_band(conf):
    if conf >= 85:
        return "High Confidence"
    elif conf >= 65:
        return "Moderate Confidence"
    else:
        return "Low Confidence"
