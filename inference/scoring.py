SCORING_RULES = {
    "choice_type_scores": {
        "Opt-in": +2.0,
        "Opt-out": +1.0,
        "Privacy controls": +0.5,
        "Dont use service/feature": 0.0,
        "Unspecified": 0.0,
    },

    "retention_period_scores": {
        "Limited": +1.5,
        "Unspecified": 0.0,
        "Indefinitely": -2.0,
    },

    "security_measure_scores": {
        "Specific": +1.5,
        "Generic": +0.5,
        "Unspecified": 0.0,
    },

    "third_party_entity_scores": {
        "Named third party": +0.5,
        "Affiliate": 0.0,
        "Public": 0.0,
        "Unspecified": 0.0,
        "Other": -0.5,
        "Unnamed third party": -1.5,
    },

    "notification_type_scores": {
        "Personal notice": +1.5,
        "General notice": +0.5,
        "Unspecified": 0.0,
        "No notification": -1.5,
    },
    "policy_change_user_choice_scores": {
        "Has choice":   +0.5,
        "None":         -0.5,
        "Unspecified":   0.0,
    },
    "access_type_scores": {
        "Delete account": +1.5,
        "Edit information": +0.5,
        "View": +0.5,
        "Other": +0.5,
        "None": -1.0,
        "Unspecified": 0.0,
    },
    "access_scope_scores": {
        "User account data": +0.5,
        "Other data": 0.0,
        "Unspecified": 0.0,
    },
    "third_party_does_scores": {
        "Does Not": +1.0,
        "Does": 0.0,
    },

    "audience_type_scores": {
        "Children": +0.5,
        "Californians": 0.0,
        "International": 0.0,
        "Other": 0.0,
    },

    "category_presence": {
        "Data Security": +1.0,
        "User Access, Edit and Deletion": +1.0,
        "User Choice/Control": +0.5,
        "Policy Change": +0.5,
        "Data Retention": 0.0,
        "First Party Collection/Use": 0.0,
        "Third Party Sharing/Collection": -0.5,
        "International and Specific Audiences": 0.0,
    },
}

THEORETICAL_MIN = -20.0
THEORETICAL_MAX=  20.0

def compute_privacy_score(segment_predictions):

    categories_seen = set()
    best_attr = {}

    for pred in segment_predictions:
        for cat, label in pred.get("category_labels", {}).items():
            if int(label) == 1:
                categories_seen.add(cat)

        for head, value in pred.get("attribute_values", {}).items():
            if head not in best_attr:
                best_attr[head]= value
            else:
                current_score = attribute_score(head, best_attr[head])
                new_score= attribute_score(head, value)
                if new_score > current_score:
                    best_attr[head] = value

    raw_score = 0.0

    for cat in categories_seen:

        raw_score += SCORING_RULES["category_presence"].get(cat, 0.0)
        if cat in ("First Party Collection/Use", "Third Party Sharing/Collection", "User Choice/Control"):
            head = f"{cat}__Choice Type"
            if head in best_attr:
                raw_score += SCORING_RULES["choice_type_scores"].get(best_attr[head], 0.0)

        head = "Data Retention__Retention Period"
        if cat == "Data Retention" and head in best_attr:
            raw_score += SCORING_RULES["retention_period_scores"].get(best_attr[head], 0.0)

        head = "Data Security__Security Measure"
        if cat == "Data Security" and head in best_attr:
            raw_score += SCORING_RULES["security_measure_scores"].get(best_attr[head], 0.0)

        head = "Third Party Sharing/Collection__Third Party Entity"
        if cat == "Third Party Sharing/Collection" and head in best_attr:
            raw_score += SCORING_RULES["third_party_entity_scores"].get(best_attr[head], 0.0)

        head = "Third Party Sharing/Collection__Does/Does Not"
        if cat == "Third Party Sharing/Collection" and head in best_attr:
            raw_score += SCORING_RULES["third_party_does_scores"].get(best_attr[head], 0.0)

        head = "Policy Change__Notification Type"
        if cat == "Policy Change" and head in best_attr:
            raw_score += SCORING_RULES["notification_type_scores"].get(best_attr[head], 0.0)


        head = "Policy Change__User Choice"
        if cat == "Policy Change" and head in best_attr:
            raw_score += SCORING_RULES["policy_change_user_choice_scores"].get(best_attr[head], 0.0)

        head = "User Access, Edit and Deletion__Access Type"
        if cat == "User Access, Edit and Deletion" and head in best_attr:
            raw_score += SCORING_RULES["access_type_scores"].get(best_attr[head], 0.0)

        head = "User Access, Edit and Deletion__Access Scope"
        if cat == "User Access, Edit and Deletion" and head in best_attr:
            raw_score += SCORING_RULES["access_scope_scores"].get(best_attr[head], 0.0)

        head = "International and Specific Audiences__Audience Type"
        if cat == "International and Specific Audiences" and head in best_attr:
            raw_score += SCORING_RULES["audience_type_scores"].get(best_attr[head], 0.0)

    normalized = ((raw_score - THEORETICAL_MIN)/ (THEORETICAL_MAX - THEORETICAL_MIN))

    score_0_10 = max(0.0, min(10.0, normalized * 10.0))

    return {
        "score_0_10":round(score_0_10, 2),
        "raw_score": round(raw_score, 3),
        "categories_seen": sorted(list(categories_seen)),
        "n_segments": len(segment_predictions),
        "best_attr":best_attr,
    }


def attribute_score(head, value):

    if "Choice Type" in head:
        return SCORING_RULES["choice_type_scores"].get(value, 0.0)

    if "Retention Period" in head:
        return SCORING_RULES["retention_period_scores"].get(value, 0.0)
    if "Security Measure" in head:
        return SCORING_RULES["security_measure_scores"].get(value, 0.0)
    if "Third Party Entity" in head:
        return SCORING_RULES["third_party_entity_scores"].get(value, 0.0)
    if "Notification Type" in head:
        return SCORING_RULES["notification_type_scores"].get(value, 0.0)
    if "Policy Change__User Choice" in head:
        return SCORING_RULES["policy_change_user_choice_scores"].get(
            value, 0.0
        )
    if "Access Type" in head:
        return SCORING_RULES["access_type_scores"].get(value, 0.0)
    if "Access Scope" in head:
        return SCORING_RULES["access_scope_scores"].get(value, 0.0)
    if "Does/Does Not" in head and "Third Party" in head:
        return SCORING_RULES["third_party_does_scores"].get(value, 0.0)
    if "Audience Type" in head:
        return SCORING_RULES["audience_type_scores"].get(value, 0.0)
    return 0.0