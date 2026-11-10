namespace PulSAR.Core;

using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System;
using System.Diagnostics;
using PulSAR.Core;

public class Classifier
{

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

    public static void ClassifyText (string text)
    {
        DebugObject dObject = new DebugObject();
        // Create the tokenizer 
        var tokenizer = BertTokenizer.Create(model_path + "vocab.txt", new BertOptions
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
        int n = tokens.Count + 2;                    // + [CLS] + [SEP]; must be <= 512
        long[] inputIds      = new long[n];  
        long[] attentionMask = new long[n];
        long[] tokenTypeIds  = new long[n];          // all 0 = one sentence

        dObject.TokenCount = n;
        

        // Map our tokens to the actual inputIDs from vocab - this lets us identify our targets for redaction later
        inputIds[0] = tokenizer.ClassificationTokenId;          // [CLS], id 101
        for (int i = 0; i < tokens.Count; i++)
            inputIds[i + 1] = tokens[i].Id;
        inputIds[n - 1] = tokenizer.SeparatorTokenId;           // [SEP], id 102

        // For each token in tokens assign an attentionMask of 1 (Real token) or 0 if its just padding
        for (int i = 0; i < n; i++)
            attentionMask[i] = 1;                    // 1 = real token, 0 = padding

        dObject.InputIDs = inputIds;
        dObject.AttentionMask = attentionMask;
        
        // Tensor 
        // build the shape of the tensor in this case 1 x tokenCount aka 'n'
        long[] shape = { 1, n };
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

        // Create an InferenceSession from the Model Path.
        // Creating and loading sessions are expensive per request.
        // They better be cached
        //InferenceSession session = new InferenceSession(model_path + "model.onnx");
        //PrintModelMetaData for model metadata/Example 1
        //PrintModelMetaData(session);

        PrintDebugInfo(dObject);
    }

    private static void PrintDebugInfo(DebugObject dObject) // [Issue] add DI to various functions like this, even though its just a helper function right now
    {
        System.Console.WriteLine(dObject.TokenCount);
        System.Console.WriteLine(dObject.CleanText);
        System.Console.WriteLine("Input Ids:");
        foreach (long iId in dObject.InputIDs)
        {
            System.Console.WriteLine($"\t{iId.ToString()}");
        }
        System.Console.WriteLine("Attention Mask:");
        foreach (long aMask in dObject.AttentionMask)
        {
            System.Console.WriteLine($"\t{aMask.ToString()}");
        }
        System.Console.WriteLine("Tensor values:");
        foreach (var item in dObject.TensorValueInputs)
        {
            System.Console.WriteLine($"{item.Key}");
        }
    }
}
