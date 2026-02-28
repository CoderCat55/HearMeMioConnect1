für meine Regen <3

* Raspiyi fişe tak
* bilgisayardan hearme ağına bağlan şifresi 123456789
* cmd yi aç
* şunu yaz cmd'ye  ssh bemol@bemolhost.local
* sonra şifre isteyecek 123456789
* sonra cmdde yeşil ve mavi yazılar görüceksin satır başında demekki bağlanmışsın
# repository silme ve tekrar yükleme
raspberyynin terminaline (şuan olduğun yer olmalı) ls yaz
ls tüm klasörleri gösterir HearMeMioConnect klasörü var mı
varsa önce şunu çalıştır  rm -rf HearMeMioConnect
yoksa devam

git clone -b {branch adı buraya gelicek} git@github.com:CoderCat55/HearMeMioConnect.git
bu olmazsa şunu dene
git clone -b {branch adı buraya gelicek} https://github.com/CoderCat55/HearMeMioConnect.git

yüklendikten sonra 
terminale mio yaz main.py'ı çalıştırmak için sonra herşey hazır

kısayollar:
rm -rf HearMeMioConnect
git clone -b yogurt_duzelt_seg_n_ca_http_pc_PCA_HTTP https://github.com/CoderCat55/HearMeMioConnect.git

# --- TEMEL SERVİS KOMUTLARI ---
sudo systemctl start hearmemio.service    # Servisi hemen şimdi başlatır
sudo systemctl stop hearmemio.service     # Servisi o an durdurur
sudo systemctl restart hearmemio.service  # Servisi kapatıp yeniden açar (ayar değişikliği sonrası)
sudo systemctl status hearmemio.service   # Servisin o anki durumunu ve hatalarını gösterir

# --- OTOMATİK BAŞLATMA AYARLARI ---
sudo systemctl enable hearmemio.service   # Pi her açıldığında servisin otomatik başlamasını sağlar
sudo systemctl disable hearmemio.service  # Pi açıldığında otomatik başlamayı kapatır

# --- HATA AYIKLAMA VE GÜNLÜKLER (LOGS) ---
journalctl -u hearmemio.service -f        # Servisin arka plan kayıtlarını (log) canlı olarak izler
journalctl -u hearmemio.service -n 50     # Servisin son 50 satırlık işlem kaydını gösterir

# --- TEMİZLİK VE SİSTEM GÜNCELLEME ---
sudo systemctl daemon-reload                 # .service dosyasında değişiklik yaparsan sistemi günceller

# --- DURUM SORGULAMA (HIZLI) ---
sudo systemctl is-active hearmemio.service   # Servis şu an çalışıyor mu? (active/inactive döner)
sudo systemctl is-enabled hearmemio.service  # Servis açılışa ekli mi? (enabled/disabled döner)