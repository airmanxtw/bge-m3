using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.HuggingFace.Tokenizer;

// https://huggingface.co/Xenova/bge-m3/tree/main

const int MaxSequenceLength = 8192;

using var options = new SessionOptions();
options.AppendExecutionProvider_CPU();
var onnxModelPath = Path.Combine(AppContext.BaseDirectory, "model_quantized.onnx");
using var session = new InferenceSession(onnxModelPath, options);

var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "tokenizer.json");
using var tokenizer = Tokenizer.FromFile(tokenizerPath);

var text = "今天天氣很好，適合出去散步。";


var embedding = CreateEmbedding(text, tokenizer, session);

// Console.WriteLine($"文字：{text}");
// Console.WriteLine($"向量維度：{embedding.Length}");
// Console.WriteLine($"向量：[{string.Join(", ", embedding.Select(value => value.ToString("G9", CultureInfo.InvariantCulture)))}]");

Console.WriteLine("========================================");
var vector1 = CreateEmbedding("林郁婷亞運奪金！全家今早10:00發500份雞排，4大超商應援優惠一次看", tokenizer, session);
var vector2 = CreateEmbedding("刀下待宰肉羊大逃亡！高雄19隻羊「集體衝靈骨塔」真相曝光", tokenizer, session);

var search = "超商優惠";

var searchVector = CreateEmbedding(search, tokenizer, session);
var similarity1 = VectorMath.CosineSimilarity(vector1, searchVector);
var similarity2 = VectorMath.CosineSimilarity(vector2, searchVector);
Console.WriteLine($"向量1與搜尋詞的相似度：{similarity1}");
Console.WriteLine($"向量2與搜尋詞的相似度：{similarity2}");


static float[] CreateEmbedding(
    string text,
    Tokenizer tokenizer,
    InferenceSession session)
{
    ArgumentException.ThrowIfNullOrWhiteSpace(text);

    var encoding = tokenizer.Encode(
        text,
        addSpecialTokens: true,
        includeAttentionMask: true).Single();

    var inputIds = encoding.Ids
        .Take(MaxSequenceLength)
        .Select(id => (long)id)
        .ToArray();

    if (encoding.Ids.Count > MaxSequenceLength)
    {
        inputIds[^1] = encoding.Ids[^1];
    }

    var attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();
    var dimensions = new[] { 1, inputIds.Length };

    var inputs = new[]
    {
        NamedOnnxValue.CreateFromTensor(
            "input_ids",
            new DenseTensor<long>(inputIds, dimensions)),
        NamedOnnxValue.CreateFromTensor(
            "attention_mask",
            new DenseTensor<long>(attentionMask, dimensions))
    };

    using var results = session.Run(inputs);
    var output = results.Single(result => result.Name == "last_hidden_state").AsTensor<float>();

    if (output.Dimensions.Length != 3 || output.Dimensions[0] != 1)
    {
        throw new InvalidOperationException(
            $"模型輸出形狀不符合預期：[{string.Join(", ", output.Dimensions.ToArray())}]");
    }

    var embeddingSize = output.Dimensions[2];
    var embedding = output.ToArray()[..embeddingSize];
    var norm = Math.Sqrt(embedding.Sum(value => value * value));

    if (norm == 0)
    {
        throw new InvalidOperationException("模型產生了零向量，無法進行正規化。");
    }

    for (var index = 0; index < embedding.Length; index++)
    {
        embedding[index] = (float)(embedding[index] / norm);
    }

    return embedding;
}

public static class VectorMath
{
    public static float CosineSimilarity(float[] first, float[] second)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);

        if (first.Length == 0 || first.Length != second.Length)
        {
            throw new ArgumentException("兩個向量必須具有相同且非零的維度。");
        }

        double dotProduct = 0;
        double firstMagnitudeSquared = 0;
        double secondMagnitudeSquared = 0;

        for (var index = 0; index < first.Length; index++)
        {
            dotProduct += first[index] * second[index];
            firstMagnitudeSquared += first[index] * first[index];
            secondMagnitudeSquared += second[index] * second[index];
        }

        if (firstMagnitudeSquared == 0 || secondMagnitudeSquared == 0)
        {
            throw new ArgumentException("無法計算零向量的 cosine similarity。");
        }

        return (float)(dotProduct / Math.Sqrt(firstMagnitudeSquared * secondMagnitudeSquared));
    }
}
