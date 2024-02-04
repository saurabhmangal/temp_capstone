# # import glob
# # data_config = [
# #     ("arxiv_sample", 30.0),
# #     ("book_sample", 40.0)
# #     #("c4", 30.0)
# # ]
# # data_dir = 'data/redpajama_sample'

# # for prefix, _ in data_config:
# #         filenames = glob.glob(f'{data_dir}/{prefix}*')
# #         print(filenames)

# #print(filenames)
        
# import json
# import tqdm
# import json

# # with open('C', 'r') as json_file:
# #     result = [json.loads(jline) for jline in json_file.read().splitlines()]

# #     json_list = list(json_file)

# # for json_str in json_list:
# #     result = json.loads(json_str)
# #     print(f"result: {result}")
# #     print(isinstance(result, dict))

# # with open('capstone_part1/data/RedPajama-Data-1T-Sample/c4_sample.jsonl','r') as f:
# #     f_list=list(f)
# #     #print(len(f_list))
# #     c=0
# #     for row in tqdm.tqdm(f_list):
# #         #print(row)
# #         # print(json.loads(row))
# #         # text = json.loads(row)["text"]
# #         # print('\n',text)
# #         if c==30606:
# #             text = json.loads(row)["text"]
# #             print(text)
# #         c+=1

# # 1

# # import json
# # import csv
# # import io

# # # # get the JSON objects from JSONL
# # with open('data/redpajama_sample/arxiv_sample.jsonl', 'r') as f:
# #     jsonl_data = f
# #     json_lines = tuple(json_line
# #                     for json_line in jsonl_data.splitlines()
# #                     if json_line.strip())
# #     jsons_objs = tuple(json.loads(json_line)
# #                     for json_line in json_lines)

# #     # write them into a CSV file
# #     fake_file = io.StringIO()
# #     print(fake_file)
# #     # writer = csv.writer(fake_file)
# #     # writer.writerow(["a", "b"])
# #     # writer.writerows((value for key, value in sorted(json_obj.items()))
# #     #                 for json_obj in jsons_objs)
# #     print(fake_file.getvalue())
# import glob
# data_config = [
#     ("arxiv_sample_00000000", 30.0),
#     ("book_sample_00000000", 40.0),
#     ("c4_sample_00000000", 30.0)
# ]
# import os
# data_dir="capstone_part1/data/redpajama_sample"
# for prefix, _ in data_config:
#     print(f"{data_dir}/{prefix}*.bin")
#     files=glob.glob(f"{data_dir}/{prefix}*.bin")
#     #files=glob.glob(f"{data_dir}/{prefix}*.bin")
#     print(len(files))
#     break
import torch
print(torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
print(num_gpus)
# Get the name and index of each GPU
gpu_info = [(torch.cuda.get_device_name(i), i) for i in range(num_gpus)]
print(gpu_info)