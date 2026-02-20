def calculate_health_risks(inputs):
    risks = {
        "Metabolic Health Risk": 0,
        "Cardiovascular Lifestyle Risk": 0,
        "Mental Well-being Risk": 0,
        "Sleep & Fatigue Risk": 0
    }

    # Age factor
    if inputs["age"] >= 40:
        risks["Metabolic Health Risk"] += 2
        risks["Cardiovascular Lifestyle Risk"] += 2

    # Physical activity
    if inputs["activity"] == "Low":
        risks["Metabolic Health Risk"] += 2
        risks["Cardiovascular Lifestyle Risk"] += 1

    # Diet pattern
    if inputs["diet"] == "Junk / Irregular":
        risks["Metabolic Health Risk"] += 2

    # Sleep duration
    if inputs["sleep"] < 6:
        risks["Sleep & Fatigue Risk"] += 3
        risks["Mental Well-being Risk"] += 1

    # Stress level
    if inputs["stress"] >= 4:
        risks["Mental Well-being Risk"] += 3

    # Smoking & alcohol
    if inputs["smoking"] != "No":
        risks["Cardiovascular Lifestyle Risk"] += 3

    if inputs["alcohol"] == "Frequent":
        risks["Mental Well-being Risk"] += 1
        risks["Metabolic Health Risk"] += 1

    # Convert score → level
    final_results = {}
    for risk, score in risks.items():
        if score <= 2:
            level = "Low"
            confidence = 0.75
        elif score <= 5:
            level = "Moderate"
            confidence = 0.82
        else:
            level = "Elevated"
            confidence = 0.90

        final_results[risk] = {
            "score": score,
            "level": level,
            "confidence": confidence
        }

    return final_results
