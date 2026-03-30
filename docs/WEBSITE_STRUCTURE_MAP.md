# 🗺️ Identix Website Structure Map

## Page Hierarchy

```
🏠 HOME (/)
├── index.html
├── Hero: "Advanced Facial Analysis Technology"
├── Features Grid:
│   ├── ✅ Facial Landmark Detection → /facial-landmarks
│   ├── ✅ Image Segmentation → /facial-landmarks
│   ├── 🔜 Deepfake Detection
│   ├── 🔜 Age & Gender Detection
│   ├── 🔜 Emotion Recognition
│   └── 🔜 Face Recognition
├── Stats Section
└── About Preview

📱 FACIAL LANDMARKS APP (/facial-landmarks)
├── landmark_index.html
├── Image Upload Mode
├── Video Upload Mode
├── Deepfake Detection Mode
├── User Authentication
└── History Tracking

ℹ️ ABOUT (/about)
├── about.html
├── Our Mission
├── Our Story
├── Core Values
├── Team
└── Technology

📧 CONTACT (/contact)
├── contact.html
├── Contact Form
├── Email: support@identix.ai
├── Phone: +1 (234) 567-890
└── Social Links

📝 BLOG (/blog)
├── blog.html
└── 6 Blog Posts:
    ├── Introducing Identix
    ├── Understanding BiSeNet
    ├── Privacy & Ethics
    ├── Building Real-Time Analysis
    ├── 99.9% Accuracy
    └── 2025 Roadmap

💼 CAREERS (/careers)
├── careers.html
└── 4 Positions:
    ├── ML Engineer
    ├── Full Stack Developer
    ├── CV Researcher
    └── UX/UI Designer

📜 LEGAL
├── /privacy-policy → privacy-policy.html
├── /terms-of-service → terms-of-service.html
├── /cookie-policy → cookie-policy.html
└── /gdpr → gdpr.html
```

## Navigation Flow

```
Landing (/) 
    ↓
    ├─→ Get Started → /facial-landmarks (App)
    ├─→ Learn More → Scroll to features
    ├─→ About Us → /about
    ├─→ Contact → /contact
    └─→ Blog → /blog

Facial Landmarks App (/facial-landmarks)
    ↓
    ├─→ Home Button → / (Back to landing)
    ├─→ Logo Click → / (Back to landing)
    ├─→ Sign In → Modal (Authentication)
    └─→ History → View past uploads

All Pages
    ↓
    ├─→ Navbar: Home, About, Contact, Blog
    ├─→ Theme Toggle: Switch dark/light mode
    └─→ Footer: All links to every page
```

## Feature Status

| Feature | Status | Route | Badge |
|---------|--------|-------|-------|
| Facial Landmark Detection | ✅ Active | /facial-landmarks | 🟢 Active Now |
| Image Segmentation | ✅ Active | /facial-landmarks | 🟢 Active Now |
| Deepfake Detection | 🔜 Coming | # | 🟠 Coming Soon |
| Age & Gender Detection | 🔜 Coming | # | 🟠 Coming Soon |
| Emotion Recognition | 🔜 Coming | # | 🟠 Coming Soon |
| Face Recognition | 🔜 Coming | # | 🟠 Coming Soon |

## API Routes (Backend)

```
Authentication:
POST /register     - Create new account
POST /login        - User login
POST /logout       - User logout
GET  /check-auth   - Check auth status

Analysis:
POST /predict            - Image landmark detection
POST /predict_video      - Video landmark detection
POST /detect_deepfake    - Deepfake analysis

User Data:
GET  /history      - Get upload history

System:
GET  /health       - Health check
```

## Images Used

| Page | Image | Source |
|------|-------|--------|
| Landing Hero | Facial Recognition Tech | Unsplash |
| Facial Landmarks Card | Face Detection | Unsplash |
| Image Segmentation Card | Technical Analysis | Unsplash |
| Deepfake Card | Security Shield | Unsplash |
| Age/Gender Card | Demographics | Unsplash |
| Emotion Card | Expressions | Unsplash |
| Face Recognition Card | Identity | Unsplash |
| About Preview | Team Working | Unsplash |

*All with placeholder fallbacks*

## Responsive Breakpoints

```css
Desktop (1024px+):
- 2-3 column layouts
- Full navigation
- Large typography

Tablet (768px - 1024px):
- 2 column layouts
- Adjusted spacing
- Medium typography

Mobile (<768px):
- 1 column layouts
- Hamburger menu (if implemented)
- Compact spacing
- Larger touch targets
```

## Theme System

```javascript
Light Mode (default):
- Background: #ffffff
- Cards: #f8f9ff
- Text: #2d3748
- Primary: #667eea

Dark Mode:
- Background: #0d1117
- Cards: #161b22
- Text: #c9d1d9 (GitHub-inspired)
- Primary: #7c8eef

Storage: localStorage['theme']
Toggle: On every page navbar
```

## Quick Links Reference

### User-Facing Pages
- `/` - Landing page
- `/facial-landmarks` - Main application
- `/about` - Company info
- `/contact` - Contact form
- `/blog` - News/articles
- `/careers` - Job listings

### Legal/Compliance
- `/privacy-policy` - Privacy information
- `/terms-of-service` - Terms & conditions
- `/cookie-policy` - Cookie usage
- `/gdpr` - GDPR compliance

### API Endpoints (No templates)
- `/predict` - Image analysis
- `/predict_video` - Video analysis
- `/detect_deepfake` - Deepfake detection
- `/login` `/register` `/logout` - Auth
- `/history` - User history
- `/check-auth` - Auth status

## File Locations

```
d:\link2\Capstone 4-1\Code_try_1\
│
├── landmark_app.py              # Flask app with all routes
│
├── templates/
│   ├── index.html               # Landing page ✅
│   ├── landmark_index.html      # Application ✅
│   ├── about.html               # About page ✅
│   ├── contact.html             # Contact page ✅
│   ├── blog.html                # Blog page ✅
│   ├── careers.html             # Careers page ✅
│   ├── privacy-policy.html      # Privacy ✅
│   ├── terms-of-service.html    # Terms ✅
│   ├── cookie-policy.html       # Cookies ✅
│   └── gdpr.html                # GDPR ✅
│
└── Documentation/
    ├── IDENTIX_GUIDE.md                    # Original guide
    ├── WEBSITE_RESTRUCTURE_COMPLETE.md     # Full documentation
    └── WEBSITE_STRUCTURE_MAP.md            # This file
```

## Component Consistency

### Every Page Has:
✅ Identix logo with fingerprint icon
✅ Navigation bar (Home, About, Contact, Blog)
✅ Theme toggle button
✅ Footer with links
✅ Consistent styling
✅ Dark/Light mode support
✅ Responsive design
✅ Font Awesome icons

### Styling Consistency:
- Border radius: 20px for cards
- Shadows: 0 10px 40px rgba(0,0,0,0.1)
- Gradients: Purple (#667eea → #764ba2)
- Hover effects: translateY(-10px)
- Transitions: all 0.3s ease
- Max width: 1400px containers

## Usage Instructions

### To Run:
```bash
cd "d:\link2\Capstone 4-1\Code_try_1"
python landmark_app.py
```

### To Test:
1. Visit http://localhost:5000/
2. Click "Try It Now" → Goes to /facial-landmarks
3. Test navigation: Home, About, Contact, Blog
4. Toggle dark/light theme
5. Check footer links to legal pages
6. Test responsive design (resize browser)

### To Update Content:
- **Images**: Replace Unsplash URLs in HTML
- **Text**: Edit HTML files directly
- **Styles**: Modify CSS in `<style>` tags
- **Routes**: Add to landmark_app.py

### To Add New Feature:
1. Create template: `templates/new-feature.html`
2. Add route in `landmark_app.py`:
   ```python
   @app.route('/new-feature')
   def new_feature():
       return render_template('new-feature.html')
   ```
3. Update landing page feature card:
   - Change badge to "Active Now"
   - Update link href="/new-feature"

## Success Metrics

✅ 10 pages created
✅ 9 routes added
✅ Professional design
✅ Real images used
✅ Fully responsive
✅ Dark mode throughout
✅ Legal compliance pages
✅ Clear navigation
✅ Scalable structure
✅ Easy to maintain

---

**Your Identix platform is ready for demo! 🎉**
