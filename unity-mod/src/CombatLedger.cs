using System;
using System.Collections.Generic;

namespace NeonFly
{
    // Pure accounting: independent of rendering, sensor rays and Unity objects.
    internal sealed class CombatLedger
    {
        private sealed class Credit { internal double damage; internal bool killed; }
        private readonly Dictionary<int, Credit> targets = new Dictionary<int, Credit>();
        private int scene = -1;
        internal double DamageFraction { get; private set; }
        internal int Kills { get; private set; }
        internal int Hits { get; private set; }

        internal void Record(int sceneHandle, int target, int before, int after, int maximum, bool owned)
        {
            if (scene != sceneHandle) { targets.Clear(); scene = sceneHandle; }
            if (!owned || maximum <= 0 || before <= 0) return;
            double loss = Math.Max(0, Math.Min(before, maximum) - Math.Max(0, Math.Min(after, maximum))) / (double)maximum;
            if (loss <= 0) return;
            Credit credit;
            if (!targets.TryGetValue(target, out credit)) targets[target] = credit = new Credit();
            double increment = Math.Min(loss, Math.Max(0, 1 - credit.damage));
            DamageFraction += increment; credit.damage += increment;
            if (increment > 0) Hits++;
            if (after <= 0 && !credit.killed) { Kills++; credit.killed = true; }
        }
    }
}
