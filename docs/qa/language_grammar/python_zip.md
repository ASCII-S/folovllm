# Python zip() 函数

## 简介

`zip()` 是 Python 内置函数，用于**将多个可迭代对象（如列表、元组）打包成元组的迭代器**。

## 基本用法

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

# zip 将两个列表配对
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# 输出：
# Alice is 25 years old
# Bob is 30 years old
# Charlie is 35 years old
```

## 工作原理

`zip()` 将对应位置的元素打包成元组：

```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
result = zip(list1, list2)

print(list(result))
# 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
```

## 在 processor.py 中的使用

```python
prompts = ["Hello", "How are you"]
sampling_params = [params1, params2]

for prompt, params in zip(prompts, sampling_params):
    request = self.process_request(prompt, params)
```

**作用**：将每个 prompt 和对应的 sampling_params 配对处理

等价于：
```python
for i in range(len(prompts)):
    prompt = prompts[i]
    params = sampling_params[i]
    request = self.process_request(prompt, params)
```

但 `zip` 更简洁优雅！

## 长度不一致时的行为

**重要**：`zip()` 以最短的可迭代对象为准：

```python
list1 = [1, 2, 3, 4]
list2 = ['a', 'b']

result = list(zip(list1, list2))
# 输出: [(1, 'a'), (2, 'b')]  ← 只有 2 对，忽略了 3 和 4
```

## 多个列表

`zip` 可以接受任意多个可迭代对象：

```python
ids = [1, 2, 3]
names = ['Alice', 'Bob', 'Charlie']
scores = [90, 85, 95]

for id, name, score in zip(ids, names, scores):
    print(f"ID: {id}, Name: {name}, Score: {score}")
```

## 解压（unzip）

使用 `*` 运算符可以解压：

```python
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)

print(numbers)  # (1, 2, 3)
print(letters)  # ('a', 'b', 'c')
```

## 优势

1. **代码简洁**：避免使用索引
2. **可读性强**：意图清晰
3. **性能好**：内存高效（返回迭代器而非列表）

## 总结

`zip(prompts, sampling_params)` = **将两个列表按位置配对，方便同时遍历**

