còn phần data: tụi e đã chia dataset 34 nhãn thành tập train/test, cái này chia train/test 7/3 cho từng nhãn rồi gộp train, gộp test lại chia 6 task với task 1,2,3,4 là 6 nhãn còn task 5,6 là 5 nhãn như sau:

✅ Task 1 — 29,592 samples
0 → BenignTraffic (5,600)
1 → DDoS-ICMP_Flood (36,554)
24 → DictionaryBruteForce (63)
25 → BrowserHijacking (30)
26 → XSS (18)
27 → Uploading_Attack (8)


✅ Task 2 — 31,435 samples
2 → DDoS-UDP_Flood (27,626)
11 → DDoS-HTTP_Flood (169)
12 → DDoS-SlowLoris (106)
13 → DoS-UDP_Flood (16,957)
29 → CommandInjection (28)
30 → Backdoor_Malware (22)


✅ Task 3 — 27,186 samples
3 → DDoS-TCP_Flood (23,149)
14 → DoS-TCP_Flood (13,630)
16 → DoS-HTTP_Flood (414)
17 → Recon-HostDiscovery (697)
18 → Recon-OSScan (517)
19 → Recon-PortScan (430)

✅ Task 4 — 23,967 samples
4 → DDoS-PSHACK_Flood (21,210)
15 → DoS-SYN_Flood (10,275)
20 → Recon-PingSweep (6)
21 → VulnerabilityScan (210)
22 → MITM-ArpSpoofing (1,614)
23 → DNS_Spoofing (925)


✅ Task 5 — 18,295 samples
5 → DDoS-SYN_Flood (20,739)
8 → DDoS-ICMP_Fragmentation (2,377)
9 → DDoS-UDP_Fragmentation (1,484)
10 → DDoS-ACK_Fragmentation (1,505)
28 → SqlInjection (31)


✅ Task 6 — 36,605 samples
6 → DDoS-RSTFINFlood (20,669)
7 → DDoS-SynonymousIP_Flood (18,189)
31 → Mirai-greeth_flood (5,016)
32 → Mirai-udpplain (4,661)
33 → Mirai-greip_flood (3,758)
