using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;

class CombatTests
{
    static void Main(string[] args)
    {
        AppDomain.CurrentDomain.AssemblyResolve += (s, e) => {
            foreach (var dir in new[] {args[1], args[2]}) {
                string file = Path.Combine(dir, new AssemblyName(e.Name).Name+".dll");
                if (File.Exists(file)) return Assembly.LoadFrom(file);
            }
            return null;
        };
        Run(args[0]);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static void Run(string path)
    {
        var assembly = Assembly.LoadFrom(path);
        var flags = BindingFlags.Instance|BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic;
        var type = assembly.GetType("NeonFly.CombatLedger");
        object ledger = Activator.CreateInstance(type, true);
        var record = type.GetMethod("Record", flags);
        Action<int,int,int,int,int,bool> hit = (scene,id,before,after,max,owned) =>
            record.Invoke(ledger, new object[] {scene,id,before,after,max,owned});
        Func<string,double> value = name => Convert.ToDouble(type.GetProperty(name, flags).GetValue(ledger, null));
        Action<double,double> equal = (a,b) => {if (Math.Abs(a-b)>1e-9) throw new Exception("Expected "+b+", got "+a);};
        hit(1, 10, 10, 8, 10, true); equal(value("DamageFraction"), .2); equal(value("Kills"), 0);
        hit(1, 10, 8, 0, 10, false); equal(value("DamageFraction"), .2); // Enemy friendly fire.
        hit(1, 10, 8, -500, 10, true); equal(value("DamageFraction"), 1); equal(value("Kills"), 1);
        hit(1, 10, -500, -501, 10, true); equal(value("DamageFraction"), 1); equal(value("Kills"), 1);
        hit(1, 10, 1, 0, 10, true); equal(value("DamageFraction"), 1); equal(value("Kills"), 1);
        hit(1, 10, 10, 5, 10, true); equal(value("DamageFraction"), 1); // Healing cannot farm damage credit.
        hit(1, 11, 20, 15, 20, true); equal(value("DamageFraction"), 1.25);
        hit(1, 12, 10, 10, 10, true); equal(value("DamageFraction"), 1.25); // Invulnerability.
        hit(1, 13, 5, 8, 10, true); equal(value("DamageFraction"), 1.25); // Healing call.
        hit(1, 14, 5, 0, 0, true); equal(value("Kills"), 1); // Invalid capacity.
        hit(2, 10, 10, 0, 10, true); equal(value("DamageFraction"), 2.25); equal(value("Kills"), 2);
        equal(value("Hits"), 4); // Partial, lethal, different target, next scene.

        int spawns=0, damageCalls=0;
        foreach (string patch in new[] {"MarkPlayerProjectiles", "CreditPlayerWeaponDamage"}) {
            var patchType=assembly.GetType("NeonFly."+patch);
            var targets=(IEnumerable<MethodBase>)patchType.GetMethod("TargetMethods", flags).Invoke(null, null);
            foreach (var target in targets) {
                var original=PatchProcessor.GetOriginalInstructions(target, (ILGenerator)null);
                var rewritten=(IEnumerable<CodeInstruction>)patchType.GetMethod("Transpiler", flags)
                    .Invoke(null, new object[] {original});
                var instructions=rewritten.ToList();
                int count=instructions.Count(i => i.operand is MethodInfo &&
                    ((MethodInfo)i.operand).DeclaringType.FullName=="NeonFly.Combat");
                if (count==0) throw new Exception("No instrumentation: "+target.Name);
                if (patch=="MarkPlayerProjectiles") spawns+=count; else damageCalls+=count;
                Console.WriteLine(target.DeclaringType.Name+"."+target.Name+": "+count+" instrumented calls");
            }
        }
        equal(spawns, 2); equal(damageCalls, 7);
        Console.WriteLine("Combat accounting and installed-game IL instrumentation checks passed.");
    }
}
