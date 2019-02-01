# ==============================================================================
# Initialize new model
# ==============================================================================

run_fgbg_model = model_zoo.bn_feature_net_skip_3D(
    receptive_field=receptive_field,
    n_features=2,
    n_frames=frames_per_batch,
    n_skips=n_skips,
    n_conv_filters=32,
    n_dense_filters=128,
    input_shape=tuple(X_test.shape[1:]),
    multires=False,
    last_only=False,
    norm_method=norm_method)
run_fgbg_model.load_weights(fgbg_weights_file)

run_conv_model = model_zoo.bn_feature_net_skip_3D(
    fgbg_model=run_fgbg_model,
    receptive_field=receptive_field,
    n_skips=n_skips,
    n_features=4,  # (background edge, interior edge, cell interior, background)
    n_frames=frames_per_batch,
    n_conv_filters=32,
    n_dense_filters=128,
    multires=False,
    last_only=False,
    input_shape=tuple(X_test.shape[1:]),
    norm_method=norm_method)
run_conv_model.load_weights(conv_weights_file)