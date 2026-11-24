namespace PulSAR.Core.Extensions;

using System;
using System.Reflection.Emit;
using System.Text;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using PulSAR.Core;
using System.Net;


public static class BertTokenizerExtensions
{
    const int numLabels = 9;
    static string[] id2label = { "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC" };
    public static List<Word> GetWords(this BertTokenizer tokenizer, int actualTokenCount, ReadOnlySpan<float> logits, IReadOnlyList<EncodedToken> tokens)
    {
        List<Word> words = new List<Word>();

        for (int t = 1; t < actualTokenCount - 1; t++)              // skip [CLS] and [SEP]
        {
            ReadOnlySpan<float> row = logits.Slice(t * numLabels, numLabels); // EXPLAIN THIS
            Span<float> softMaxRow = new Span<float>(row.ToArray()); //make this 9 as the number of labels
            System.Numerics.Tensors.TensorPrimitives.SoftMax(row, softMaxRow);

            int best = 0;
            for (int k = 1; k < numLabels; k++) {
                if (row[k] > row[best]) best = k;
                if (softMaxRow[k] > softMaxRow[best]) best = k;
            }

            EncodedToken tok = tokens[t - 1];        // shift back past [CLS]

            LabelId testLabel = (LabelId)best;
            

            if (!tok.Value.StartsWith("##"))
            {
                words.Add(new Word()
                {
                    startIndex = tok.Offset.Start.Value,
                    endIndex = tok.Offset.End.Value,
                    bestLabel = new Core.Label
                    {
                        label = (LabelId) best,
                        probability = softMaxRow[best]
                    },
                    tokens = new List<WordToken>()
                    {
                        new WordToken()
                        {
                            startIndex = tok.Offset.Start.Value,
                            endIndex = tok.Offset.End.Value,
                            token = tok.Value,
                            label = new Core.Label
                            {
                                label = (LabelId) best,
                                probability = softMaxRow[best]
                            }
                        }
                    },
                    wordBuilder = new StringBuilder(tok.Value)
                });
            }
            else // this will need to accomodatelastordefault style checking or throw an exception saying an error with input
            {
                var w = words.Last();
                Core.Label tokenLabel = new Core.Label
                {
                    label = (LabelId) best,
                    probability = softMaxRow[best]
                };

                words.Last()
                    .wordBuilder.Append(tok.Value.Replace("##", ""));
                
                words.Last().tokens.Add(new WordToken()
                        {
                            startIndex = tok.Offset.Start.Value,
                            endIndex = tok.Offset.End.Value,
                            token = tok.Value,
                            label = tokenLabel
                        });

                if (words.Last().bestLabel.probability < softMaxRow[best] && (LabelId)best != LabelId.NONE)
                {
                    words.Last().bestLabel = tokenLabel;
                }
                words.Last().endIndex = tok.Offset.End.Value;
            }
        }

        return words;
    }
}
