namespace PulSAR.Core;

using Microsoft.ML.OnnxRuntime;

public class DebugObject
{
    public int TokenCount {get; set;}

    public string CleanText {get; set;}

    public long[] InputIDs {get; set;}

    public long[] AttentionMask {get; set;}

    public Dictionary<string, OrtValue> TensorValueInputs {get; set;}

}