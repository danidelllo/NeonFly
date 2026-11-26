using System;
using System.IO;
using System.Reflection;

// Exercises the actual compiled bridge's clipping code without starting Unity.
class RangeClipTests
{
    static MethodInfo clip;
    static int checks;
    static void Check(float x, float y, float dx, float dy, float range,
                      bool expected, float start = 0, float end = 0)
    {
        object[] args = { x, y, dx, dy, range, 0f, 0f };
        bool actual = (bool)clip.Invoke(null, args);
        if (actual != expected || (actual &&
            (Math.Abs((float)args[5]-start) > 1e-4f || Math.Abs((float)args[6]-end) > 1e-4f)))
            throw new Exception("Unexpected clipped interval: " + String.Join(", ", args));
        checks++;
    }
    static void Main(string[] args)
    {
        AppDomain.CurrentDomain.AssemblyResolve += (s, e) => {
            string path = Path.Combine(args[1], new AssemblyName(e.Name).Name + ".dll");
            return File.Exists(path) ? Assembly.LoadFrom(path) : null;
        };
        var assembly = Assembly.LoadFrom(args[0]);
        clip = assembly.GetType("NeonFly.RangeSensor").GetMethod("ClipViewport", BindingFlags.Static | BindingFlags.NonPublic);
        Check(.5f, .5f, .1f, 0, 12, true, 0, 5); // Inside: previous behavior retained.
        Check(.5f, 1.01f, 0, -.1f, 12, true, .1f, 10.1f); // Ceiling regression.
        Check(.5f, 1.01f, 0, .1f, 12, false); // Points away from scene.
        Check(.5f, 1.01f, .1f, 0, 12, false); // Parallel, outside.
        Check(.5f, -.01f, 0, .1f, 12, true, .1f, 10.1f);
        Check(-.01f, .5f, .1f, 0, 12, true, .1f, 10.1f);
        Check(1.01f, .5f, -.1f, 0, 12, true, .1f, 10.1f);
        Check(-.1f, 1.1f, .1f, -.1f, 12, true, 1, 11);
        Check(-.1f, -.1f, .1f, -.1f, 12, false); // No corner overlap.
        Check(.5f, 1.01f, 0, -.1f, .05f, false); // Visible entry beyond range.
        Check(.5f, 1, 0, -.1f, 12, true, 0, 10);
        Check(.5f, 1, 0, .1f, 12, false); // Zero-length tangent.
        // Many positions, including all boundaries: returned segments stay
        // inside the viewport and never exceed the original distance budget.
        var random = new Random(784);
        for (int i = 0; i < 10000; i++) {
            float x = (float)random.NextDouble()*3-1, y = (float)random.NextDouble()*3-1;
            float angle = (float)random.NextDouble()*6.2831853f;
            float dx = (float)Math.Cos(angle)*.1f, dy = (float)Math.Sin(angle)*.1f;
            object[] input = {x, y, dx, dy, 12f, 0f, 0f};
            bool hit = (bool)clip.Invoke(null, input);
            float a = (float)input[5], b = (float)input[6];
            if (hit) {
                foreach (float t in new float[] {a, (a+b)/2, b})
                    if (t < 0 || t > 12 || x+dx*t < -1e-5 || x+dx*t > 1.00001 ||
                        y+dy*t < -1e-5 || y+dy*t > 1.00001)
                        throw new Exception("Segment escaped viewport/range");
            }
            for (int j = 0; j <= 120; j++) {
                float t = j/10f, px = x+dx*t, py = y+dy*t;
                if (px > 1e-5 && px < .99999 && py > 1e-5 && py < .99999 &&
                    (!hit || t < a-1e-4 || t > b+1e-4))
                    throw new Exception("Missed visible ray segment");
            }
            checks++;
        }
        var angleMethod=assembly.GetType("NeonFly.RangeSensor").GetMethod("RayAngle", BindingFlags.Static | BindingFlags.NonPublic);
        int[] counts = {0,0,0,0};
        double previous = -999;
        for (int i=0; i<126; i++) {
            double angle=(float)angleMethod.Invoke(null, new object[] {i})*180/Math.PI;
            if (angle <= previous || angle <= -45 || angle >= 315) throw new Exception("Invalid/duplicate ray angle");
            int group=angle<45?0:angle<135?1:angle<225?2:3;counts[group]++;
            if (i>0 && i!=55 && i!=63 && i!=71) {
                double expected=group==0||group==3?90.0/55:11.25;
                if (Math.Abs(angle-previous-expected)>1e-4) throw new Exception("Wrong sector spacing");
            }
            previous=angle;
        }
        if (counts[0]!=55 || counts[1]!=8 || counts[2]!=8 || counts[3]!=55) throw new Exception("Wrong ray groups");
        Console.WriteLine("Passed " + checks + " range clipping cases and 126-ray sector layout checks.");
    }
}
