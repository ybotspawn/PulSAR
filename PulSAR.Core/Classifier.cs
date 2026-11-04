namespace PulSAR.Core;

using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System;

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
        // Create the tokenizer 
        var tokenizer = BertTokenizer.Create(model_path + "vocab.txt", new BertOptions
        {
            LowerCaseBeforeTokenization = false
        });

        // Clean our text up:
        // [Issue] - Make this its own method
        string clean = text.Replace('\r', ' ').Replace('\n', ' ').Replace('\t', ' ');

        // Take our text and tokenize them 
        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(clean, out var normalized);

        // Lets take a look at that token count.
        // [Issue] - Create a mechanism to chunk the inputs so we never bust the limit of our proposed model 510
        System.Console.WriteLine(tokens.Count);

        // Create an InferenceSession from the Model Path.
        // Creating and loading sessions are expensive per request.
        // They better be cached
        //InferenceSession session = new InferenceSession(model_path + "model.onnx");
        //PrintModelMetaData for model metadata/Example 1
        //PrintModelMetaData(session);

        
    }
}
