import argparse
import yaml
from pathlib import Path

def update_yaml_file(section):
    # 定义文件路径
    yaml_file_path = Path("./alpaca_eval/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_turbo_fn/configs.yaml")

    # 读取 YAML 文件
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # 更新字段
    config['alpaca_eval_gpt4_turbo_fn']['prompt_template'] = f"alpaca_eval_gpt4_turbo_fn/alpaca_eval_fn_{section}.txt"

    # 写回 YAML 文件
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(config, file)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Update the section in the YAML configuration file.")
    parser.add_argument('--section', type=str, required=True, help="The section to be used in the file name.")
    args = parser.parse_args()

    # 更新 YAML 文件
    update_yaml_file(args.section)

if __name__ == "__main__":
    main()
