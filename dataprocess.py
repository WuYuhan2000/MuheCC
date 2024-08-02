##使用refactoringMiner提取重构信息
# import subprocess
# import pandas as pd
# import os
#
# # 设置 GitHub OAuth token
# github_oauth_token = "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"
# os.environ['GITHUB_OAUTH'] = github_oauth_token
#
# # 读取 Excel 文件
# file_path = "MuheCC_Dataset.xlsx"
# df = pd.read_excel(file_path)
#
# # 提取 commit_id 和 project 列
# commit_ids = df['commit_id']
# projects = df['project']
#
# # 创建或清空输出文件
# output_json_path = f"/Users/apple/Downloads/output{commit_ids}.json"
# with open(output_json_path, 'w') as f:
#     f.write('')
#
# # 处理每个 commit_id 和 project
# for commit_id, project in zip(commit_ids, projects):
#     github_repo_url = f"https://github.com/{project}"
#     command = [
#         './RefactoringMiner',
#         '-gc', github_repo_url,
#         commit_id,
#         '1000',
#         '-json', output_json_path
#     ]
#
#     try:
#         result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"Processed commit {commit_id} in project {project}: {result.stdout.decode('utf-8')}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error processing commit {commit_id} in project {project}: {e.stderr.decode('utf-8')}")
#
# print("All commits processed.")

#统计数量
# import os
# import json
#
# # 输出目录
# output_dir = "output"
#
# # 初始化计数器
# empty_commits_count = 0
# empty_refactorings_count = 0
# have = 0
# # 遍历输出目录中的所有JSON文件
# for json_file in os.listdir(output_dir):
#     if json_file.endswith(".json"):
#         json_path = os.path.join(output_dir, json_file)
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#             if not data['commits']:
#                 empty_commits_count += 1
#             elif all(not commit.get('refactorings') for commit in data['commits']):
#                 empty_refactorings_count += 1
#             else:
#                 have += 1
#
# # 打印统计结果
# print(f"Number of JSON files with empty commits: {empty_commits_count}")
# print(f"Number of JSON files with empty refactorings: {empty_refactorings_count}")
# print(have)

#将重构信息保存为一列新数据，对于没有提取到的保存为空
# import os
# import json
# import pandas as pd
#
# #读取Excel文件
input_file = 'MuheCC_Dataset.xlsx'
df = pd.read_excel(input_file)

# 定义输出目录
output_dir = './output'

# 初始化一个新的列
df['refactoring'] = ''

# 遍历每一行，根据commit_id查找对应的JSON文件
for index, row in df.iterrows():
    commit_id = row['commit_id']
    # project = row['project']
    json_file = os.path.join(output_dir, f'{commit_id}_{index+1}.json')

    refactoring_text = ''

    # 检查JSON文件是否存在
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if data['commits']:
                refactorings = data['commits'][0].get('refactorings', [])
                if refactorings:
                    # 将refactorings转换为字符串表示
                    refactoring_text = json.dumps(refactorings)

    # 如果refactoring_text为空，使用comment_diff的值
    if not refactoring_text:
            refactoring_text = row['comment_diff']

    # 将refactoring_text填入新的列中
    df.at[index, 'refactoring'] = refactoring_text

# 保存新的Excel文件
output_file = 'MuheCC_Dataset_with_refactoring.xlsx'
df.to_excel(output_file, index=False)


