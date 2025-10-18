# python import
import os
import logging
from typing import Dict, Union
# package import
import pandas as pd
# local import

__all__ = ['read_metadata_as_df']

logger = logging.getLogger(__name__)


def read_metadata_as_df(data_dir: str, primary_key: str,
                        parser: Dict[str, Union[str, tuple[Union[list[str], str], str]]],
                        group_by: Union[list[str], None] = None,
                        explode: Union[list[str], None] = None,
                        drop: Union[list[str], None] = None) -> pd.DataFrame:
    assert primary_key in parser.keys(), (f'Please add {primary_key} in your parser, using a lambda function to choose '
                                          f'which file is needed in your folder: {data_dir}.')
    metadata = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_metadata = {}
            input_path = os.path.join(root, file)

            # parse image files, invalid file skipped
            primary_eval_parser = eval(parser[primary_key])
            if not primary_eval_parser(input_path):
                continue
            else:
                file_metadata[primary_key] = input_path

            # parse metadata
            for key, single_parser in parser.items():
                if key == primary_key:
                    continue
                else:
                    if isinstance(single_parser, str):
                        eval_parser = eval(single_parser)
                        lambda_input = (input_path,)
                    elif isinstance(single_parser, tuple) or isinstance(single_parser, list):
                        assert len(single_parser) == 2, "Parser tuple should have length 2: (existing_keys, lambda function)."
                        existing_keys, eval_parser = single_parser[0], eval(single_parser[1])
                        existing_keys = existing_keys if isinstance(existing_keys, list) else [existing_keys]
                        assert all([k in file_metadata.keys() for k in existing_keys]), \
                            f"Existing keys {existing_keys} not found in file metadata for parsing {key}."
                        lambda_input = (file_metadata[k] for k in existing_keys)
                    else:
                        raise ValueError("Parser should be either a string or a tuple of (existing_keys, lambda function).")
                    if not callable(eval_parser):
                        raise ValueError(f"Parser for {key} should be a lambda function.")
                    file_metadata[key] = eval_parser(*lambda_input)
            metadata.append(file_metadata)

    assert len(metadata) > 0, f"No valid files found in {data_dir}, please check the path or the function"
    logger.info(f"Found {len(metadata)} valid files in {data_dir}")
    meta_df = pd.DataFrame(metadata)

    if group_by:
        assert all([col in meta_df.columns for col in group_by]), f"Columns {group_by} not found in metadata"
        assert set(group_by) != set(meta_df.columns), "group_by columns should not be all columns"
        assert primary_key not in group_by, "primary_key column should not be in group_by columns"

        meta_df = meta_df.groupby(group_by)[primary_key].apply(list).reset_index()
        logger.info(f"Grouped metadata by {group_by}")

    if explode:
        assert all([col in meta_df.columns for col in explode]), f"Columns {explode} not found in metadata"
        meta_df = meta_df.explode(explode).reset_index(drop=True)
        logger.info(f"Exploded metadata columns {explode}")

    if drop:
        assert all([col in meta_df.columns for col in drop]), f"Columns {drop} not found in metadata"
        meta_df = meta_df.drop(columns=drop)
        logger.info(f"Dropped metadata columns {drop}")
    return meta_df

if __name__ == "__main__":
    # example usage
    example_dir = "."
    data_col_name = "py_files"
    function_parser = {
        "py_files": "lambda x: x.endswith('.py')",
        "name": "lambda x: os.path.basename(x).split('.')[0]",
        'module': "lambda path: os.path.dirname(path)",
        "load": (["name", "py_files"], "lambda x, y: x.startswith('load') and y.endswith('.py')"),
    }

    df = read_metadata_as_df(example_dir, data_col_name, function_parser, None)
    print(df.head().to_string())

    # if one col is varied in groups, the final df will not contain this col
    df = read_metadata_as_df(example_dir, data_col_name, function_parser, ["module", 'load'])
    print(df.head().to_string())

    df = read_metadata_as_df(example_dir, data_col_name, function_parser, ["module", 'load'], explode=['py_files'])
    print(df.head().to_string())

    df = read_metadata_as_df(example_dir, data_col_name, function_parser, None, drop=['load'])
    print(df.head().to_string())
    