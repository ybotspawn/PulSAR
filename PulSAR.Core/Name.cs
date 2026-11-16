namespace PulSAR.Core;

public class Name
{
    public Word? Title {get; set;}
    public List<Word> properNames {get; set;}
    public int? StartIndex => properNames.FirstOrDefault()?.startIndex;
    public int? EndIndex => properNames.LastOrDefault()?.endIndex;
}