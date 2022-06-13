# from click.testing import CliRunner
#
# from src.models.train_test_split import train_test_split
#
# runner = CliRunner()
#
#
# def test_cli_command():
#     result = runner.invoke(
#         train_test_split,
#         [
#             "data/processed/unsplit_train_data/",
#             "both",
#             "data/processed/train_data_both",
#         ],
#     )
#     assert result.exit_code == 0
