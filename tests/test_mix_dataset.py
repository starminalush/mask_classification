# from click.testing import CliRunner
#
# from src.data.mix_datasets import mix_datasets
#
# runner = CliRunner()
#
#
# def test_cli_command():
#     result = runner.invoke(
#         mix_datasets,
#         [
#             "data/raw/raw_data/",
#             "data/processed/cropped_images/",
#             "data/processed/unsplit_train_data/",
#         ],
#     )
#     assert result.exit_code == 0
