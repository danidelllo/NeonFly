using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Security.Cryptography;
using HarmonyLib;
using BepInEx;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NeonFly
{
    [BepInPlugin("local.neonfly.bridge", "NeonFly bridge", "0.8.0")]
    public sealed class Bridge : BaseUnityPlugin
    {
        private const int MaxImage = 32 * 1024 * 1024;
        private readonly object gate = new object();
        private TcpListener listener;
        private volatile bool stopping;
        private Request pending;
        private readonly byte[] key = new byte[32];
        private Harmony patches;
        private bool controlled;
        private float lastControl;
        private int resetCompletions;

        private sealed class Request
        {
            public int Op;
            public bool Complete;
            public string Meta;
            public byte[] Image;
            public bool Started;
            public int Direction, Ticks, Handle, Deaths, Completions;
            public int ExpectedHandle, ExpectedDeaths, ExpectedCompletions;
            public bool IsStep { get { return Op == 5 || Op == 7 || Op == 8 || Op == 9; } }
        }

        [Serializable]
        private sealed class Status
        {
            public bool ok = true;
            public string scene;
            public int frame;
            public float timeScale;
            public int width, height;
            public int sceneHandle, deaths, completions, appliedTicks;
            public int fixedUpdates;
            public double physicsSeconds, deathSeconds, completionSeconds;
            public string controller;
            public bool gameOver;
            public float x, y;
            public string[] buttons;
            public string[] bindings;
            public int bullets, rockets, shipActionCount;
            public int health, maxHealth, maxBullets, maxRockets, money;
            public double damageFraction;
            public bool controlled;
            public string bridgeVersion = "0.8.0";
            public bool sensorValid, sensorInsideView;
            public float sensorRange, sensorVx, sensorVy, sensorScreenX, sensorScreenY;
            public float[] sensorDistance, sensorVisible, sensorVisibleStart, sensorGeometryDistance;
            public int[] sensorKind, sensorGeometryKind;
        }

        private void Awake()
        {
            Application.runInBackground = true;
            using (var rng = RandomNumberGenerator.Create()) rng.GetBytes(key);
            var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NeonFly");
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, "session.key"), key);
            patches = new Harmony("local.neonfly.bridge");
            patches.PatchAll(typeof(Bridge).Assembly);
            listener = new TcpListener(IPAddress.Loopback, 47651);
            listener.Start(1);
            new Thread(Serve) { IsBackground = true }.Start();
            StartCoroutine(Pump());
            Logger.LogInfo("Observation bridge listening on 127.0.0.1:47651");
        }

        private IEnumerator Pump()
        {
            var end = new WaitForEndOfFrame();
            while (!stopping)
            {
                yield return end;
                Request r;
                lock (gate) { r = pending; if (r == null || r.Complete) {
                    GameAdapter.Remaining = 0;
                    if (controlled && Time.realtimeSinceStartup - lastControl > 30) {
                        controlled = false; Time.timeScale = 1;
                    }
                    continue;
                } }
                Texture2D tex = null;
                bool waiting = false;
                try
                {
                    var scene = SceneManager.GetActiveScene();
                    if (r.Op >= 3 && !r.Started)
                    {
                        r.Started = true;
                        r.Handle = scene.handle;
                        r.Deaths = GameAdapter.Deaths;
                        r.Completions = GameAdapter.Completions;
                        if (r.Op == 9 || r.Op == 10) {
                            controlled = true;
                            lastControl = Time.realtimeSinceStartup;
                            Time.timeScale = 1;
                        }
                        if (r.IsStep && r.Op >= 7 && (r.ExpectedDeaths != r.Deaths ||
                            r.ExpectedCompletions != r.Completions))
                        {
                            GameAdapter.Applied = 0;
                            GameAdapter.Remaining = 0;
                        }
                        else if (r.IsStep && r.Op >= 7 && r.ExpectedHandle != scene.handle)
                            throw new InvalidOperationException("episode_scene_changed");
                        else if (r.IsStep) GameAdapter.Step(r.Direction, r.Ticks, r.Op >= 8);
                        else if (r.Op == 10) {
                            if (scene.name != "Level_1" && GameAdapter.Completions <= resetCompletions)
                                throw new InvalidOperationException("training_reset_requires_Level_1");
                            resetCompletions = GameAdapter.Completions;
                            SceneManager.LoadScene("Level_1");
                        }
                        else if (r.Op == 6) GameAdapter.Reset();
                        else GameAdapter.Click(r.Op);
                        waiting = true;
                        continue;
                    }
                    if ((r.IsStep && GameAdapter.Driving && GameAdapter.Ship.isActiveAndEnabled &&
                        GameAdapter.Deaths == r.Deaths && GameAdapter.Completions == r.Completions) ||
                        ((r.Op == 6 || r.Op == 10 || r.Op == 3 || r.Op == 4) && scene.handle == r.Handle))
                    { waiting = true; continue; }
                    GameAdapter.Remaining = 0;
                    var ship = FindObjectOfType<Player>();
                    var rick = FindObjectOfType<RickController>();
                    Component actor = rick && rick.isActiveAndEnabled ? (Component)rick :
                        ship && ship.isActiveAndEnabled ? (Component)ship : null;
                    var manager = FindObjectOfType<GameSessionManager>();
                    var ammo = ship ? ship.GetComponent<PlayerAmmo>() : null;
                    var upgrade = ship ? ship.GetComponent<PlayerRocketUpgrade>() : null;
                    var health = ship ? ship.GetComponent<PlayerHealth>() : null;
                    if (controlled) { Time.timeScale = 0; lastControl = Time.realtimeSinceStartup; }
                    var scan = RangeSensor.Capture(ship);
                    var status = new Status { scene = SceneManager.GetActiveScene().name,
                        sensorValid = scan.valid, sensorInsideView = scan.insideView, sensorRange = scan.range,
                        sensorDistance = scan.distance, sensorVisible = scan.visible,
                        sensorVisibleStart = scan.visibleStart,
                        sensorKind = scan.kind, sensorGeometryDistance = scan.geometryDistance,
                        sensorGeometryKind = scan.geometryKind, sensorVx = scan.vx, sensorVy = scan.vy,
                        sensorScreenX = scan.screenX, sensorScreenY = scan.screenY,
                        sceneHandle = scene.handle, frame = Time.frameCount, timeScale = Time.timeScale,
                        controller = actor == null ? "none" : actor.GetType().Name,
                        gameOver = manager && manager.IsGameOver, deaths = GameAdapter.Deaths,
                        completions = GameAdapter.Completions, appliedTicks = r.IsStep ? GameAdapter.Applied : 0,
                        fixedUpdates = fixedUpdates,
                        physicsSeconds = Time.fixedTimeAsDouble, deathSeconds = GameAdapter.DeathSeconds,
                        completionSeconds = GameAdapter.CompletionSeconds,
                        shipActionCount = 36, bindings = r.Op == 1 ? GameAdapter.Bindings() : null,
                        bullets = ammo ? ammo.CurrentBullets : -1,
                        rockets = ammo && upgrade ? ammo.GetCurrentRockets(upgrade.SelectedRocketType) : -1,
                        maxRockets = ammo && upgrade ? ammo.GetMaxRockets(upgrade.SelectedRocketType) : -1,
                        maxBullets = ammo ? ammo.MaxBullets : -1,
                        health = health ? health.CurrentHealth : 0, maxHealth = health ? health.MaxHealth : -1,
                        money = MoneyManager.Instance ? MoneyManager.Instance.GetMoney() : -1,
                        damageFraction = GameAdapter.DamageFraction, controlled = controlled,
                        x = actor ? actor.transform.position.x : 0, y = actor ? actor.transform.position.y : 0,
                        buttons = r.Op == 1 ? GameAdapter.Buttons() : null };
                    if (r.Op == 2 || r.IsStep || r.Op == 6 || r.Op == 10)
                    {
                        if ((long)Screen.width * Screen.height > 16777216)
                            throw new InvalidOperationException("Frame exceeds capture limit.");
                        tex = ScreenCapture.CaptureScreenshotAsTexture();
                        status.width = tex.width;
                        status.height = tex.height;
                        r.Image = tex.EncodeToPNG();
                        if (r.Image == null || r.Image.Length == 0 || r.Image.Length > MaxImage)
                            throw new InvalidOperationException("Invalid PNG size.");
                    }
                    r.Meta = JsonUtility.ToJson(status);
                }
                catch (Exception e)
                {
                    Logger.LogError(e);
                    r.Meta = "{\"ok\":false,\"error\":\"observation_failed\"}";
                    r.Image = null;
                    if (controlled) { controlled = false; Time.timeScale = 1; }
                }
                finally
                {
                    if (tex != null) Destroy(tex);
                    if (!waiting) lock (gate) { r.Complete = true; Monitor.PulseAll(gate); }
                }
            }
        }

        private void Serve()
        {
            while (!stopping)
            {
                try
                {
                    using (var client = listener.AcceptTcpClient())
                    {
                        client.ReceiveTimeout = 3000;
                        client.SendTimeout = 5000;
                        using (var s = client.GetStream()) Handle(s);
                    }
                }
                catch (SocketException e) { if (!stopping) Logger.LogWarning(e.Message); }
                catch (IOException e) { if (!stopping) Logger.LogWarning(e.Message); }
                catch (ObjectDisposedException) { if (!stopping) throw; }
            }
        }

        private void Handle(NetworkStream s)
        {
            var header = new byte[5];
            for (int i = 0; i < header.Length;)
            {
                int n = s.Read(header, i, header.Length - i);
                if (n == 0) return;
                i += n;
            }
            string meta = "{\"ok\":false,\"error\":\"unsupported_request\"}";
            byte[] image = null;
            if (Encoding.ASCII.GetString(header, 0, 4) == "NFL1" && header[4] >= 1 && header[4] <= 10)
            {
                var r = new Request { Op = header[4] };
                if (r.Op >= 3)
                {
                    var credential = new byte[key.Length];
                    for (int i = 0; i < credential.Length;)
                    {
                        int n = s.Read(credential, i, credential.Length - i);
                        if (n == 0) return;
                        i += n;
                    }
                    int mismatch = 0;
                    for (int i = 0; i < key.Length; i++) mismatch |= key[i] ^ credential[i];
                    if (mismatch != 0) throw new IOException("Invalid session credential.");
                    if (r.IsStep)
                    {
                        using (var reader = new BinaryReader(s, Encoding.UTF8, true))
                        {
                            r.Direction = reader.ReadByte() - (r.Op >= 8 ? 0 : 1);
                            r.Ticks = reader.ReadByte();
                            if (r.Op >= 7)
                            {
                                r.ExpectedHandle = reader.ReadInt32();
                                r.ExpectedDeaths = reader.ReadInt32();
                                r.ExpectedCompletions = reader.ReadInt32();
                            }
                        }
                    }
                }
                lock (gate)
                {
                    if (stopping) return;
                    pending = r;
                    var timer = System.Diagnostics.Stopwatch.StartNew();
                    while (!r.Complete && !stopping)
                    {
                        long left = 5000 - timer.ElapsedMilliseconds;
                        if (left <= 0) break;
                        Monitor.Wait(gate, (int)left);
                    }
                    pending = null;
                    if (stopping) return;
                    if (r.Complete) { meta = r.Meta; image = r.Image; }
                    else meta = "{\"ok\":false,\"error\":\"render_timeout\"}";
                }
            }
            var json = Encoding.UTF8.GetBytes(meta);
            using (var w = new BinaryWriter(s))
            {
                w.Write(Encoding.ASCII.GetBytes("NFL1"));
                w.Write(json.Length);
                w.Write(image == null ? 0 : image.Length);
                w.Write(json);
                if (image != null) w.Write(image);
            }
        }

        private int fixedUpdates;
        private void FixedUpdate() { fixedUpdates++; }

        private void OnDestroy()
        {
            stopping = true;
            lock (gate) Monitor.PulseAll(gate);
            if (listener != null) listener.Stop();
            if (patches != null) patches.UnpatchSelf();
            if (controlled) Time.timeScale = 1;
        }
    }
}
