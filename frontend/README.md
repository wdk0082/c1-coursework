# 5D Regression Frontend

A modern Next.js frontend for the 5D to 1D regression API platform.

## Features

- **Dashboard**: Real-time health check and system status
- **Dataset Management**: Upload and manage .npz datasets
- **Model Training**: Configure and train neural network models with full control over:
  - Architecture (hidden layers, activation, dropout)
  - Training parameters (learning rate, batch size, epochs, etc.)
  - Data processing (split ratios, standardization, missing value strategies)
- **Predictions**: Make single or batch predictions using trained models
- **Model Management**: View, compare, and manage trained models

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client for API communication

## Prerequisites

- Node.js 18.0 or higher
- npm or yarn package manager
- Backend API running on http://localhost:8000

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env.local
```

4. Update `.env.local` if your backend is running on a different URL:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Development Mode

Start the development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000

### Production Build

Build for production:
```bash
npm run build
```

Start the production server:
```bash
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── page.tsx           # Dashboard
│   │   ├── upload/            # Dataset upload
│   │   ├── train/             # Model training
│   │   ├── predict/           # Predictions
│   │   ├── datasets/          # Dataset management
│   │   ├── models/            # Model management
│   │   │   └── [id]/          # Model details
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   └── Navigation.tsx     # Navigation bar
│   └── lib/                   # Utilities
│       └── api/               # API client
│           ├── types.ts       # TypeScript types
│           └── client.ts      # API functions
├── public/                    # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
```

## Usage

### 1. Upload a Dataset

1. Navigate to **Upload Dataset**
2. Select a `.npz` file containing your data
3. The file must have `X` (5D features) and `y` (1D targets)
4. Click **Upload Dataset**

### 2. Train a Model

1. Navigate to **Train Model**
2. Select an uploaded dataset
3. Configure:
   - **Architecture**: Hidden layers, activation function, dropout
   - **Training Parameters**: Learning rate, batch size, epochs, etc.
   - **Data Processing**: Split ratios, standardization, missing value handling
4. Click **Start Training**
5. Wait for training to complete (progress shown in backend terminal)

### 3. Make Predictions

1. Navigate to **Predict**
2. Select a trained model
3. Choose input type:
   - **Single Input**: Enter 5 values manually
   - **Multiple Inputs**: Paste multiple vectors (one per line)
4. Click **Make Prediction**
5. View results

### 4. Manage Datasets and Models

- **Datasets**: View all uploaded datasets, see statistics, and train models
- **Models**: View all trained models, compare metrics, and view detailed configurations

## API Integration

The frontend communicates with the backend API using the following endpoints:

- `GET /health` - System health check
- `POST /upload` - Upload dataset
- `POST /train` - Train model
- `POST /predict` - Make predictions
- `GET /datasets` - List datasets
- `GET /models` - List models
- `GET /models/{id}` - Get model details

All API calls are handled through the `api` client in `src/lib/api/client.ts`.

## Configuration Options

### Missing Value Strategies

- **ignore**: Remove rows with missing values
- **mean**: Fill with column mean
- **median**: Fill with column median
- **zero**: Fill with zeros
- **forward_fill**: Forward fill missing values

### Activation Functions

- **relu**: Rectified Linear Unit (default)
- **tanh**: Hyperbolic Tangent
- **sigmoid**: Sigmoid function

## Development

### Type Safety

All API responses are fully typed using TypeScript interfaces defined in `src/lib/api/types.ts`.

### Styling

The project uses Tailwind CSS with custom utility classes defined in `globals.css`:
- `.btn-primary` - Primary action buttons
- `.btn-secondary` - Secondary action buttons
- `.card` - Card container
- `.input-field` - Form inputs
- `.label` - Form labels

### Adding New Features

1. Define TypeScript types in `src/lib/api/types.ts`
2. Add API client functions in `src/lib/api/client.ts`
3. Create page components in `src/app/`
4. Update navigation in `src/components/Navigation.tsx`

## Troubleshooting

### Cannot connect to backend

- Ensure the backend API is running on http://localhost:8000
- Check the `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS settings in the backend if running on different ports

### Build errors

- Delete `.next` folder and `node_modules`
- Run `npm install` again
- Check for TypeScript errors with `npm run lint`

## License

Academic project for Cambridge C1 coursework.
