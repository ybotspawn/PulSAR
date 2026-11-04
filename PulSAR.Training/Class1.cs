namespace PulSAR.Training;

// Except for some of the above deprecated API classes, nearly all of C# API classes are IDisposable. Meaning they need to be disposed after use, otherwise you will get memory leaks. Because OrtValues are used to hold tensor data, the sizes of the leaks can be huge. They are likely to accumulate with each Run call, as each inference call requires input OrtValues and returns output OrtValues. Do not hold your breath for finalizers which are not guaranteed to ever run, and if they do, they do it when it is too late.

public class Class1
{
    public Class1()
    {
        
    }
}
