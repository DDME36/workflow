# GitHub Actions Setup Guide

## วิธีตั้งค่า GitHub Actions สำหรับ Fear & Greed Alert

### 1. Push โค้ดขึ้น GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fear_greed_alert.git
git push -u origin main
```

### 2. ตั้งค่า Secrets

ไปที่ Repository Settings > Secrets and variables > Actions > New repository secret

เพิ่ม Secret:
- **Name:** `DISCORD_WEBHOOK_URL`
- **Value:** `https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN`

### 3. เปิดใช้งาน Actions

ไปที่ tab "Actions" ใน repository แล้วกด "I understand my workflows, go ahead and enable them"

### 4. ทดสอบ Manual Run

1. ไปที่ Actions tab
2. เลือก "Daily Fear & Greed Check"
3. กด "Run workflow"
4. เลือก branch และกด "Run workflow"

### 5. Schedule

Workflow จะรันอัตโนมัติ:
- **Daily Check:** ทุกวันจันทร์-ศุกร์ เวลา 05:00 น. (เวลาไทย)
- **Weekly Retrain:** ทุกวันอาทิตย์ (optional)

### 6. ดู Logs

ไปที่ Actions tab > เลือก workflow run > ดู logs

### Troubleshooting

**Q: Workflow ไม่รัน?**
- ตรวจสอบว่า Actions เปิดใช้งานแล้ว
- ตรวจสอบ cron syntax
- ดู logs ใน Actions tab

**Q: Discord alert ไม่ส่ง?**
- ตรวจสอบ Secret `DISCORD_WEBHOOK_URL`
- ทดสอบ webhook URL ด้วย curl

**Q: Model ไม่โหลด?**
- ต้อง train model ก่อน: `python train_v2.py`
- Commit และ push folder `models/`

### Files

```
.github/
└── workflows/
    └── daily_check.yml    # Main workflow
```

### Cron Schedule Reference

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
│ │ │ │ │
* * * * *

# Examples:
0 22 * * 1-5    # 22:00 UTC Mon-Fri = 05:00 ICT
0 14 * * 1-5    # 14:00 UTC Mon-Fri = 21:00 ICT (after US market close)
```
