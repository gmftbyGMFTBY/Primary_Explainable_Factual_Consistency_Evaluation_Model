# Explainable Factual Consistency Evalualuation Model

本项目包含一些稍微有点意思但是没啥卵用的代码
在该项目中，通过在8卡3090上使用OLoRA微调InternLM-7B模型
使得InternLM-7B模型具有比较意思的能力：
该模型可以对给定的Claim和Context进行分析，识别出Claim和Context不一致的部分（存在矛盾的部分），并将其具体不一致的矛盾内容的span生成出来，同时生成对应的简单的自然语言解释，最终，根据生成的不一致的span和简单的自然语言解释，生成最终的评估分数（0 for entailment, 1 for neutral, 2 for contradiction）

### 如何使用

参考`generate_data/README.md`准备好训练数据

```bash
./scripts/train.sh
```

如何测试

```bash
# delta_model_path你可以写成自己训练后的save path
# 也可以用我上传好的模型参数: https://huggingface.co/GMFTBY/ExplainableEvaluation
./scripts/test.sh
```
