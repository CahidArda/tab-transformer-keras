
def get_X_from_groups(feature_set, groups):
    """
    Generate input sets from a feature set for use in my custom TabTransformer
    Args:
        feature_set(:obj:`DataFrame`): Pandas dataframe to generate input sets from.
        groups(:obj:`list`): List of lists with each member list denoting a group of features.
    Returns:
        List of datasets corresponding to the groups
    """

    result = []
    for group in groups:
        result.append(feature_set[group])
    return result

def get_X_from_features(feature_set, cont_features, cat_features):
    groups = [cont_features]
    groups.extend(cat_features)
    return get_X_from_groups(feature_set, groups)