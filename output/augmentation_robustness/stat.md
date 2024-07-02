A látens robosztusságának tesztelésére megfigyelhetjük, hogy a módosított adatok a látens térben milyen változásokon esnek át.
A kísérlet során a CIFAR10 teszt adatainek reprezentációit figyeltük meg. 8 adatseten teszteltünk:
 - 1 referencia (nincs változás)
 - 1 soft (a tanítás során ismert augemntációkat alkalmazzuk
 - 1 soft++ (a tanítás során ismert, de felerősített augmentációk)
 - 5 hard (a képet drasztikusan megváltoztató, mint pl. vágás, ismeretlen augmentációk)

 Vizsgált értékek: hossz, elmozdulás, cos sim (halmaz és osztály szerint), illetve a veszteségek (kld, mse, class, az egész halmazra vonatkozólag)

 Eredmények: A látens téren való elváltoztatások gyengék, ha tanult augmentációk, és észlelhetőek, hogyha nem, és jelentős hatással van a képekre. A teljesen felismerhetetlen képeket félreklasszifikálja, a látens térben nem különülnek el, az eloszlás pedig nem a standardból származik. Minél inkább ismerős a modell számára, annál közelebb vagyunk a referencia értékekhez.

 ![Ref](./augmentation_robustness/20240702135028/lspace_0_labels.png)

 A fenti a referencia látens vektortér dim redukciót követve, a címkék szerint színezve

  ![Soft](./augmentation_robustness/20240702135028/lspace_2_labels.png)
  
Ez a módosított, soft++ augmentációt tartalmazó vektortér

![Hard](./augmentation_robustness/20240702135028/lspace_4_labels.png)

A 4. képen már a hard, ismeretlen augmentáció szerepel (perspektívikus változtatás), itt a kép felismerése statisztikailag sikerült.

![Hard++](./augmentation_robustness/20240702135028/lspace_6_labels.png)

A 6. képen is a hard, ismeretlen augmentáció szerepel, viszont ez tipikus példája annak a változatnak, amikor a modell már nem ismeri fel a képeket, és nem képes rekonstruálni. A térben nem választhatóak szét szépen az osztályok, az osztályozásuk romlik, stb.

Az elmondható, hogy a veszteségek szórás szigorúan növekszik a bonyolúltabb augmentációknál, az osztályok szórása pedig növekszik és egyre jobban eltér a standardtól. A cos similarity viselkedését viszont nem tudtam még kellően leírni.

![KLD](./augmentation_robustness/20240702135028/avg_kld.png)

![Diff](./augmentation_robustness/20240702135028/avg_latent_space_diff_v_s.png)

A mérték nem állandó, a szórás viszont növekszik

![Vector size](./augmentation_robustness/20240702135028/avg_ls_v_size.png)

Az osztályvektorok egyre távoliabbak

![Cos sim](./augmentation_robustness/20240702135028/avg_cos_sim.png)

A cos similarity viselkedését nem sikerült megmagyarázzam egyelőre.

