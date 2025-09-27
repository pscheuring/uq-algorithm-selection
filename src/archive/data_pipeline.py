# def sample_data(config):
#     data_cfg = config["data"]
#     train_data_cfg = sampling_cfg["train"]
#     test_data_cfg = sampling_cfg["test"]

#     # Choose sampler
#     sampler_type = sampling_cfg["sampler"]

#     if sampler_type == "single":
#         sampler = CubicSingleFeatureSampler(seed=sampling_cfg["seed"])
#         train_df = sampler.sample_train_data(
#             n_unique=train_data_cfg["n_unique"],
#             n_repeats=train_data_cfg["n_repeats"],
#             min_val=train_data_cfg["min_val"],
#             max_val=train_data_cfg["max_val"],
#             n_sparse_center=train_data_cfg["n_sparse_center"],
#             use_sparse_center=train_data_cfg["use_sparse_center"],
#         )
#         test_df = sampler.sample_test_data(
#             n_points=test_data_cfg["n_points"],
#             min_val=test_data_cfg["min_val"],
#             max_val=test_data_cfg["max_val"],
#         )

#     elif sampler_type == "x3":
#         sampler = x3Sampler(
#             n_samples=sampling_cfg["n_samples"], seed=sampling_cfg["seed"]
#         )
#         x_min = sampling_cfg["x_min"]
#         x_max = sampling_cfg["x_max"]
#         train_df = sampler.sample_train_data(x_min=x_min, x_max=x_max, train=True)
#         test_df = sampler.sample_test_data(x_min=-7, x_max=7, train=False)
#         test_df["aleatoric_true"] = 3.0

#     elif sampler_type == "linear":
#         sampler = LinearMultiFeatureSampler(
#             n_features=sampling_cfg["n_features"], seed=sampling_cfg["seed"]
#         )
#         train_df = sampler.sample_train_data(
#             n_unique=train_data_cfg["n_unique"],
#             n_repeats=train_data_cfg["n_repeats"],
#             min_val=train_data_cfg["min_val"],
#             max_val=train_data_cfg["max_val"],
#             sparse_center_frac=train_data_cfg["sparse_center_frac"],
#         )
#         test_df = sampler.sample_test_grid(
#             grid_length=test_data_cfg["grid_length"],
#             min_val=test_data_cfg["min_val"],
#             max_val=test_data_cfg["max_val"],
#         )
#     else:
#         raise ValueError(f"Unknown sampler type: {sampler_type}")

#     return train_df, test_df
