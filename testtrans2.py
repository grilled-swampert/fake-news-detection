# import transformers
# from transformers import TrainingArguments

# print("Transformers version:", transformers.__version__)
# print("TrainingArguments file:", TrainingArguments.__module__)
# print("TrainingArguments file path:", transformers.training_args.__file__)

# with open(transformers.training_args.__file__, 'r', encoding='utf-8') as f:
#     lines = [next(f) for _ in range(20)]
# print("".join(lines))



import inspect
from transformers import TrainingArguments

print("Transformers version:", TrainingArguments.__module__)
print("TrainingArguments source file:", inspect.getfile(TrainingArguments))

# Get first ~40 lines of the TrainingArguments class definition
source = inspect.getsource(TrainingArguments)
for i, line in enumerate(source.splitlines()[:40], start=1):
    print(f"{i:03d}: {line}")
