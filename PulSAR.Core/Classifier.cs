namespace PulSAR.Core;

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
using PulSAR.Core.Extensions;
using System.Net;

public class Classifier
{

    public static string[] id2label = { "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC" };
    public const int numLabels = 9;
    public static int internalDebug  = 0;
    public static string ybot = "My ybot library is functioning";
    private static string model_path = "/Users/mattwood/repos/git/ybotspawn/PulSAR/PulSAR.models/";
    public Classifier ()
    {
        
    }

    private static void PrintModelMetaData(InferenceSession session)
    {
        // A good first exercise for #2: load the model into an InferenceSession and print InputMetadata and OutputMetadata. You should see three int64 inputs (input_ids, attention_mask, token_type_ids) and one logits output of shape [batch, tokens, 9]. Check that against Netron. If the names differ, use whatever the model reports.
        foreach (var kv in session.InputMetadata)
            Console.WriteLine($"IN  {kv.Key}: {kv.Value.ElementType.Name} [{string.Join(", ", kv.Value.Dimensions)}]");

        foreach (var kv in session.OutputMetadata)
            Console.WriteLine($"OUT {kv.Key}: {kv.Value.ElementType.Name} [{string.Join(", ", kv.Value.Dimensions)}]");;

    }

    public static void ClassifyText (string text, int debug_enabled = 0)
    {
        internalDebug = debug_enabled;

        DebugObject dObject = new DebugObject();
        // Create the tokenizer 
        var tokenizer = BertTokenizer.Create($"{model_path}vocab.txt", new BertOptions
        {
            LowerCaseBeforeTokenization = false
        });

        // Clean our text up:
        // [Issue] - Make this its own method
        string clean = text.Replace('\r', ' ').Replace('\n', ' ').Replace('\t', ' ');
        dObject.CleanText = clean;
        // Take our text and tokenize and encode them --- Encoding returns both the token and its text value.  It doesn't include the CLS/SEP which we need to add later
        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(clean, out var normalized);

        // [Issue] - Create a mechanism to chunk the inputs so we never bust the limit of our proposed model 510
        int actualTokenCount = tokens.Count + 2;                    // + [CLS] + [SEP]; must be <= 512
        long[] inputIds      = new long[actualTokenCount];  
        long[] attentionMask = new long[actualTokenCount];
        long[] tokenTypeIds  = new long[actualTokenCount];          // all 0 = one sentence

        dObject.TokenCount = actualTokenCount;
        

        // Map our tokens to the actual inputIDs from vocab - this lets us identify our targets for redaction later
        inputIds[0] = tokenizer.ClassificationTokenId;          // [CLS], id 101
        for (int i = 0; i < tokens.Count; i++)
            inputIds[i + 1] = tokens[i].Id;
        inputIds[actualTokenCount - 1] = tokenizer.SeparatorTokenId;           // [SEP], id 102

        // For each token in tokens assign an attentionMask of 1 (Real token) or 0 if its just padding
        for (int i = 0; i < actualTokenCount; i++)
            attentionMask[i] = 1;                    // 1 = real token, 0 = padding

        dObject.InputIDs = inputIds;
        dObject.AttentionMask = attentionMask;
        
        // Tensor 
        // build the shape of the tensor in this case 1 x tokenCount aka 'actualTokenCount'
        long[] shape = { 1, actualTokenCount };
        // Creating the tensor based on the values of the array sizing above, which in turn is derived from the data provided to us (clean text)
        using var idsValue  = OrtValue.CreateTensorValueFromMemory(inputIds, shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(attentionMask, shape);
        using var typeValue = OrtValue.CreateTensorValueFromMemory(tokenTypeIds, shape);

        // Dictionary of tensorValues
        var inputs = new Dictionary<string, OrtValue>
        {
            ["input_ids"]      = idsValue,
            ["attention_mask"] = maskValue,
            ["token_type_ids"] = typeValue,
        };
        
        dObject.TensorValueInputs = inputs;

        // Build the inference session
        using var session    = new InferenceSession($"{model_path}model.onnx");  // create once and reuse in real code
        
        // Basic empty run options - why just because we might do stuff with it later
        using var runOptions = new RunOptions();

        // Get the results of the inference session with our inputs and session Outputnames
        using var results    = session.Run(runOptions, inputs, session.OutputNames);

        // Logits show the scoring before we characterize them as an actual percentage
        ReadOnlySpan<float> logits = results[0].GetTensorDataAsSpan<float>();

        DebugPrint("Logits");
        foreach (var item in logits)
        {
            DebugPrint($"\t{item.ToString()}");
        }

        List<Word> words = GetWords(actualTokenCount, logits, tokens);
        List<Word> extensionWords = tokenizer.GetWords(actualTokenCount, logits, tokens);

        List<Name> names = PopulateNames(extensionWords);

        System.Console.WriteLine(names.Count);
        foreach (var n in names)
        {
            string fullName = "";
            foreach (var w in n.properNames)
            {
                if (w.wordBuilder.ToString().Equals("'"))
                {
                    fullName = fullName.Trim();
                    fullName = $"{fullName}{w.wordBuilder.ToString()}";
                }
                else
                    fullName = $"{fullName}{w.wordBuilder.ToString()} ";
            }
            fullName = fullName.Trim();
            System.Console.WriteLine($"Full proper name: {fullName} --- Start: {n.StartIndex} -> {n.EndIndex}");
        }
        
        //names.ForEach(n => n.properNames.ForEach(w =>System.Console.WriteLine($"Full Proper Name: {w.wordBuilder.ToString()} --- Start: {n.StartIndex} -> {n.EndIndex}")));
        // PrintDebugInfo(dObject);
        System.Console.WriteLine($"Actual Text: {text}");
    }   

    private static void DebugPrint(string text)
    {
        if (internalDebug == 1)
        {
            System.Console.WriteLine(text);
        }
    }

    private static List<Name> PopulateNames(List<Word> words)
    {
        List<Name> documentOrderedNames = new List<Name>();
        System.Console.WriteLine("Words");
        foreach (var w in words)
        {
            if (w.bestLabel.label == LabelId.BPER)
            {
                System.Console.WriteLine($"\t{w.wordBuilder.ToString()} --- {(LabelId)w.bestLabel.label} --- {w.bestLabel.probability} --- Start: {w.startIndex} -> {w.endIndex}");
                documentOrderedNames.Add(new Name()
                {
                    properNames = new List<Word>(){ w }
                });
            }
            else if (w.bestLabel.label == LabelId.IPER)
            {
                System.Console.WriteLine($"\t{w.wordBuilder.ToString()} --- {(LabelId)w.bestLabel.label} --- {w.bestLabel.probability} --- Start: {w.startIndex} -> {w.endIndex}");
                documentOrderedNames.Last()?.properNames.Add(w);
            }
        }
        return documentOrderedNames;
    }

    private static void GetSoftMaxValues(ReadOnlySpan<float> row, int numLabels)
    {
        Span<float> softMaxRow = new Span<float>(row.ToArray()); //make this 9 as the number of labels
        System.Numerics.Tensors.TensorPrimitives.SoftMax(row, softMaxRow);

        int best = 0;
        for (int k = 1; k < numLabels; k++)
            if (softMaxRow[k] > softMaxRow[best]) best = k;

        System.Console.WriteLine($"Label: {best} Confidence: {softMaxRow[best]}");
        // numLabels where labels is 9 labels, --- From id2Label array
        // best = the best label that applies; 
        // B-PER is Label 3
    }

    private static void PrintDebugInfo(DebugObject dObject) // [Issue] add DI to various functions like this, even though its just a helper function right now
    {
        DebugPrint(dObject.TokenCount.ToString());
        DebugPrint(dObject.CleanText);
        DebugPrint("Input Ids:");
        foreach (long iId in dObject.InputIDs)
        {
            DebugPrint($"\t{iId.ToString()}");
        }
        // System.Console.WriteLine("Attention Mask:");
        // foreach (long aMask in dObject.AttentionMask)
        // {
        //     System.Console.WriteLine($"\t{aMask.ToString()}");
        // }
        // System.Console.WriteLine("Tensor values:");
        // foreach (var item in dObject.TensorValueInputs)
        // {
        //     System.Console.WriteLine($"{item.Key} - {item.Value.ToString()}"); // not really useful, but just leaving it here for now
        // }

        
    }
}
