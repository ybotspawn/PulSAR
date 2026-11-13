using System.Text;

namespace PulSAR.Core;

public class Word
{
    public int startIndex {get; set;}
    public int endIndex {get; set;}

    public StringBuilder wordBuilder {get; set;}

    public Label bestLabel {get; set;}

    public List<WordToken> tokens {get; set;}

}