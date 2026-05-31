rm(list = ls())
cat("\f")

library(keras)
library(tensorflow)
library(rBayesianOptimization)
library(forecast)
library(zoo)
library(ggplot2)
library(plotly)
library(scales)
library(patchwork)
library(tidyr)

#========================
# Settings
#========================

# ARDL(2,3), NARDL(2,1,5)
# بسبب الفروق والتأخيرات يبدأ النموذج الهجين من المشاهدة السابعة

y_lag <- 2
x_lag <- 5
max_lag <- 6

seed <- 22
set.seed(seed)
tensorflow::set_random_seed(seed)

#========================
# Dates
#========================

dates <- seq(
  from = as.yearmon("2008-01"),
  to   = as.yearmon("2025-12"),
  by   = 1/12
)

dates <- as.Date(dates, frac = 1)

# تواريخ العينة المستخدمة في النموذج الهجين
dates_model <- dates[(max_lag + 1):length(dates)]

head(dates_model)
tail(dates_model)


X <- c(83.86, 81.68, 90.48, 95.68, 104.28, 115.98,
       123.73, 113.05, 101.74, 83.70, 59.03, 43.97,
       34.88, 37.14, 37.83, 43.85, 48.93, 56.30,
       64.49, 64.16, 68.16, 65.74, 71.50, 74.02,73.37,73.99,72.98,76.46,79.93,73.32,71.07,72.08,71.65,73.25,
       77.59,80.74,86.44,91.09,97.75,107.84,114.77,108.34,105.32,109.21,
       104.71,104.95,104.42,106.97,106.05,109.13,113.73,118.47,116.41,102.66,
       90.59,97.5,107.05,108.12,105.21,104.86,104.1,105.38,108.11,103,
       98,97,97.41,102.93,104.77,104.87,102.99,102.62,104.61,101.99,
       102.5,101.15,100.55,101.3,102.61,102.33,98.71,92.58,83.41,71.19,
       58.4,44.88,47.65,48.8,51.78,55.8,56.23,51.02,43.87,42.19,
       40.7,37.57,31.46,24.76,23.38,27.28,33.48,36.06,39.39,39.65,
       38.89,39.47,41.37,41.79,45.03,48.59,48.88,47.89,46.62,46.6,
       43.8,43.34,45.18,48.73,52.21,55.77,58.68,61.14,61.13,60.06,
       63.16,67.07,69.19,68.49,69.5,72.51,73.42,64.71,55.44,55.95,
       59.59,63.22,66.85,66.85908333,61.52553226,60.57219328,58.08187879,58.59287603,57.27,58.8,
       62.02,60.67,53.47,37.16,19.28,19.21,31.69,39.88011475,42.93928767,41.06,
       39.08166667,40.90103659,46.95,52.00801515,57.52,62.08,62.66671795,64.19285882,68.38,70.89,
       69.82,70.92,76.1496875,77.44,74.55,78.39,89.06,102.91,107.07,107.73,
       112.22,106.73,96.87,91.85,88,84,76.87,74.81,75.64,74.5,
       76.39,73.71,71.28,76.18,82.21,88.66,88.3,84,78.5,76.5,
       77.2,79.5,82.6,81.9,80.6,81.3,77.3,72.15,70.88,70.77,
       70.8,74,75.19,72.83,68.37,63.47,65.73,68.02,68.33,67.87,
       64.53,62.02)




Y<-c(0.966047, 7.206042, 3.708015, 1.847428,
        5.784292, 3.298239, 4.444174, 1.582540,
        -0.163472, 0.646186, -2.207128, -6.263556,
        0.455635, -0.089697, 0.345976, 0.181277,
        0.203305, -0.142095, 0.800733, 2.535169,
        1.887459, -0.219606, -0.129397, -3.186431 ,1.167092,0.591717,3.200308,2.432613,-0.299351,0.555582,-0.638341,-0.601925,2.264676,0.78262,
          1.083018,-5.368876,3.753475,3.55471,4.003632,3.467064,4.845268,2.330218,1.493167,2.634611,
          3.36578,2.084202,3.044725,-4.217599,4.911688,0.366409,4.028115,7.868429,4.447886,0.913304,
          -0.429816,-2.200963,-33.486497,43.486236,3.699126,-4.512297,5.31212,3.636617,2.97935,1.589834,
          0.885671,-0.998703,0.580767,0.440993,1.114979,-1.485558,0.793779,-7.955481,6.160454,4.430141,
          2.664308,5.296009,4.43657,4.047466,1.599736,7.248677,2.724697,1.621915,0.326732,-18.726308,
          3.591843,0.726395,-0.857814,-0.579764,-0.31165,0.2488,-2.734974,1.138311,0.859061,-0.619896,
          -2.88786,-2.499715,-0.631863,-1.973342,-2.102281,-0.302638,-1.354561,-1.547769,-0.4035,0.239763,
          -1.356254,-0.473074,0.124925,-2.877573,0.54966,0.016599,1.262062,0.653788,0.844388,0.471688,
          -0.629568,-1.523169,2.401039,-1.08988,-0.737483,-0.373284,2.987158,2.095138,3.306126,1.997049,
          1.27022,3.224101,0.314064,2.459039,1.611791,6.196045,0.151023,0.084891,1.161275,0.472898,
          0.39444,1.278161,0.675258,3.262589,2.065295,-0.77282,-1.749654,-7.640502,7.172602,-10.47607,
          1.594581,0.461953,-1.266026,-0.976126,-1.662536,-1.423537,-5.592662,-2.118432,-2.310161,-1.478469,
          -1.828481,3.717142,-1.824906,-0.423616,2.116775,1.604735,-0.688448,1.011841,-0.293473,1.131305,
          7.304891,-4.383611,1.564892,-0.88858,4.339339,3.359636,3.709668,3.958399,7.617522,3.460658,
          7.269388,4.899997,5.429033,5.528069,10.228691,-15.062545,1.278121,3.367641,-0.077741,0.700367,
          0.93066,0.30867,3.844941,-0.073025,7.311819,-1.391169,-4.686522,-18.268132,2.750908,1.15202,
          2.242783,-0.563673,1.294405,0.791178,-4.041025,1.289065,14.555185,-17.53196,-3.047269,-8.644857,
          -1.035246,0.495075,-0.350497,-0.155157,0.223679,6.113871,-10.590058,0.197609,-3.028247,-2.972762,
          -1.780106,-4.16085)

df <- data.frame(
  Date = dates,
  Oil_Price = X,
  Surplus_Deficit = Y
)

#========================
# Mean Line
#========================

mean_oil <- mean(df$Oil_Price, na.rm = TRUE)

#========================
# Professional Oil Plot
#========================

ggplot(df, aes(x = Date, y = Oil_Price)) +
  
  geom_line(
    color = "#1F3A5F",
    linewidth = 1.35,
    lineend = "round"
  ) +
  
  geom_hline(
    yintercept = mean_oil,
    color = "#A23E48",
    linewidth = 0.9,
    linetype = "dashed"
  ) +
  
  annotate(
    "label",
    x = max(df$Date),
    y = mean_oil,
    label = paste0("Mean = ", round(mean_oil, 2), " USD"),
    hjust = 1,
    vjust = -0.7,
    size = 4.5,
    fontface = "bold",
    color = "#A23E48",
    fill = "white",
    label.size = 0
  ) +
  
  labs(
    title = "Monthly Oil Prices",
    x = NULL,
    y = "USD per Barrel"
  ) +
  
  scale_x_date(
    date_breaks = "2 years",
    date_labels = "%Y",
    expand = expansion(mult = c(0.01, 0.04))
  ) +
  
  theme_minimal(base_size = 15) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 21,
      hjust = 0.5,
      color = "#1F2937"
    ),
    
    axis.title.y = element_text(
      face = "bold",
      size = 13,
      color = "#374151"
    ),
    
    axis.text = element_text(
      size = 11,
      color = "#374151"
    ),
    
    panel.grid.major = element_line(
      color = "#E5E7EB",
      linewidth = 0.45
    ),
    
    panel.grid.minor = element_blank(),
    
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.background = element_rect(
      fill = "white",
      color = NA
    )
  )


#========================
# Professional Budget Plot
#========================


mean_budget <- mean(df$Surplus_Deficit, na.rm = TRUE)

ggplot(df, aes(x = Date, y = Surplus_Deficit)) +
  
  # Area Shadow
  geom_area(
    fill = "#D6EAF8",
    alpha = 0.45
  ) +
  
  # Main Line
  geom_line(
    color = "#1B4F72",
    linewidth = 1.4
  ) +
  
  # Zero Line
  geom_hline(
    yintercept = 0,
    color = "#2E2E2E",
    linewidth = 0.9
  ) +
  
  # Mean Line
  geom_hline(
    yintercept = mean_budget,
    color = "#C1121F",
    linewidth = 1.1,
    linetype = "dashed"
  ) +
  
  # Mean Text
  annotate(
    "text",
    x = max(df$Date),
    y = mean_budget + 1.5,
    label = paste0(
      "Mean = ",
      round(mean_budget, 2)
    ),
    color = "#C1121F",
    hjust = 1,
    size = 5,
    fontface = "bold"
  ) +
  
  labs(
    title = "Budget Surplus / Deficit",
    subtitle = "Monthly Fiscal Position of Iraq",
    x = NULL,
    y = "Trillion Iraqi Dinars"
  ) +
  
  scale_x_date(
    date_breaks = "2 years",
    date_labels = "%Y"
  ) +
  
  theme_minimal(base_size = 15) +
  
  theme(
    
    plot.title = element_text(
      size = 22,
      face = "bold",
      hjust = 0.5,
      color = "#1B2631"
    ),
    
    plot.subtitle = element_text(
      size = 13,
      hjust = 0.5,
      color = "#5D6D7E"
    ),
    
    axis.title.y = element_text(
      size = 14,
      face = "bold",
      color = "#1B2631"
    ),
    
    axis.text = element_text(
      size = 11,
      color = "#2E4053"
    ),
    
    panel.grid.major = element_line(
      color = "#E5E7E9",
      linewidth = 0.5
    ),
    
    panel.grid.minor = element_blank(),
    
    plot.background = element_rect(
      fill = "#F8F9F9",
      color = NA
    ),
    
    panel.background = element_rect(
      fill = "#FFFFFF",
      color = NA
    )
  )

#========================
# Combine Two Plots
#========================


#------------------------
# Oil Plot
#------------------------

p1 <- ggplot(df, aes(x = Date, y = Oil_Price)) +
  
  geom_line(
    color = "#1F3A5F",
    linewidth = 1.1
  ) +
  
  geom_hline(
    yintercept = mean_oil,
    color = "#A23E48",
    linewidth = 0.8,
    linetype = "dashed"
  ) +
  
  labs(
    title = "Oil Prices",
    x = NULL,
    y = "USD"
  ) +
  
  theme_minimal(base_size = 11) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      size = 14,
      color = "#1F2937"
    ),
    
    axis.title.y = element_text(
      face = "bold",
      size = 10
    ),
    
    axis.text = element_text(
      size = 9,
      color = "#374151"
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.grid.major = element_line(
      color = "#E5E7EB",
      linewidth = 0.4
    ),
    
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.background = element_rect(
      fill = "white",
      color = NA
    )
  )

#------------------------
# Budget Plot
#------------------------

p2 <- ggplot(df, aes(x = Date, y = Surplus_Deficit)) +
  
  geom_line(
    color = "#145A32",
    linewidth = 1.1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "#2E2E2E",
    linewidth = 0.7
  ) +
  
  geom_hline(
    yintercept = mean_budget,
    color = "#C1121F",
    linewidth = 0.8,
    linetype = "dashed"
  ) +
  
  labs(
    title = "Budget Surplus / Deficit",
    x = NULL,
    y = "Trillion IQD"
  ) +
  
  theme_minimal(base_size = 11) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      size = 14,
      color = "#1B2631"
    ),
    
    axis.title.y = element_text(
      face = "bold",
      size = 10
    ),
    
    axis.text = element_text(
      size = 9,
      color = "#374151"
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.grid.major = element_line(
      color = "#E5E7EB",
      linewidth = 0.4
    ),
    
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.background = element_rect(
      fill = "white",
      color = NA
    )
  )

#========================
# Side-by-Side Layout
#========================

p1 + p2 +
  plot_layout(ncol = 2)




#Data Monthly
#========================================
# ARDL fitted values
# يبدأ من اللاك الرابع
#========================================

  
YHAT_ARDL <- c(
  rep(NA,3),
  -0.693214241, 4.396112515, 2.821443005, 4.624747419,
  5.066229829, 4.701757151, -0.182217616, -1.91764086,
  -3.767542639, -4.33850296, -4.075070627, -3.742203659,
  -0.375970212, -0.561817193, 1.431219442, 1.858912233,
  2.236642421, 2.004809079, -0.522488615, 2.135454649,
  0.781863625, 4.368952015, 2.322883555, 0.818774498,
  0.379829964, -0.084790027, 2.114772749, 2.299311387,
  -0.161451192, 1.117334357, 0.883993243, 0.81494064,
  1.688890956, 5.548077105, 2.82798126, 2.383168531,
  2.587550673, 3.751860327, 4.024918901, 4.088492869,
  1.4054705, 1.601555738, 3.103171199, 1.101149276,
  2.362072892, 4.988712015, 3.528324997, 3.036820173,
  3.705629341, 1.744031897, 1.891721936, 1.616325403,
  -0.146793382, 1.44523792, 20.42129653, -5.867308674,
  -5.49541333, 4.286451774, 2.378722401, 1.266180947,
  2.121722199, 3.074444273, 1.382906961, 2.243215197,
  3.27899245, 3.478914107, 4.731362986, 4.673550567,
  3.720022624, 6.572146522, 2.390919271, 1.233965586,
  0.91878606, 0.957972085, 0.34458317, 0.9237997,
  2.398078274, 0.420600615, 0.550110643, 0.408957542,
  -0.282389213, 6.436436075, -1.353876291, -4.521184725,
  -3.850908232, 0.938114394, 0.568209697, 0.83521939,
  2.096575663, -0.309054487, -2.596995969, -2.703522761,
  -0.198836807, -0.249228411, -1.970541461, -2.726994202,
  -2.529984847, -1.572772673, 0.149963753, 1.38767084,
  0.01445796, -0.225809242, -0.43814456, -0.709452804,
  -0.751241888, 1.012662335, -0.107391085, 0.305757081,
  -0.000388335, -0.905369015, -1.232118225, -1.380684954,
  -0.500390789, -0.626484216, -1.339531256, 0.251171261,
  1.482019971, 1.469285306, 0.195258162, -0.171635762,
  -0.635131149, -0.774832278, -0.341843284, 0.245219081,
  1.390326724, 0.663113725, 0.009199705, -1.287145798,
  0.552730812, 0.737107816, -2.41178, -2.529161409,
  0.442925071, 1.192837847, 1.347092017, 0.148514289,
  -0.792529811, -1.072293004, 1.083121251, 3.22295359,
  -1.092019128, 3.326172851, 2.163081203, 0.530899194,
  -0.643383091, -3.071612692, -5.551906568, -5.736886919,
  1.118340156, 4.224121605, 2.595206472, 0.891319241,
  -0.367691012, -2.430668894, 0.199446075, 2.208544028,
  0.93904822, 0.945205366, 1.978837541, 0.855375395,
  1.490213826, 1.869605259, -1.34426672, 1.800463825,
  2.075922171, 3.088476754, 0.541968, -0.650261962,
  1.981255732, 4.035633724, 3.663477654, 2.52595524,
  0.388659547, 1.337065525, -1.46276315, -3.225525192,
  -4.288794091, 5.495282342, 1.239096802, -1.765212226,
  1.546459192, 1.612489495, 0.227238371, 1.564374052,
  0.257035726, 0.85800594, 3.259024433, 2.502691758,
  2.429222911, 0.494020974, 0.100729001, 0.215069195,
  1.247291157, 1.682230736, 1.723556547, 1.860161053,
  0.808330529, 0.796932745, 1.302696194, -0.304515378,
  -0.159190926, 1.088175958, 1.01915655, 0.887072465,
  1.749847463, 0.668330667, -0.284311984, -0.181090156,
  -0.19713154, 1.76983577, 0.97415831, 0.236053813,
  0.373146297
)

YHAT_NARDL <- c(
  rep(NA,6),
  1.181872718, 1.289646964, 1.789370176, -0.674232054,
  -6.800072113, -3.211588272, -4.685762847, -5.072207086,
  0.426042305, -0.683697328, 0.821867705, -2.394154559,
  -1.623601385, 0.067377397, -1.470645321, -0.971162294,
  -0.540763572, 0.656385436, 1.989541487, 2.314094322,
  0.684164106, -0.90024109, -0.449959692, 0.247403497,
  1.572431169, 0.370835473, -0.863785376, 1.893754582,
  0.767239447, 0.625232611, 3.685300563, 1.416503297,
  -0.089628823, -0.110703559, 1.558331274, 1.711702651,
  2.891012014, 2.243720012, 0.357835486, 4.339142996,
  3.555551852, 1.112213592, 7.734658741, 2.902730675,
  3.448132707, 2.762885899, 1.844743159, 0.185984151,
  1.719740859, 0.364486708, -1.731193936, 21.66103026,
  -3.721655077, -4.875414927, 5.785397647, 2.269708871,
  2.86498242, 2.322049546, 3.276456065, 3.02545393,
  2.15320394, 3.550631147, 5.866663224, 4.882705073,
  5.245833838, 4.905270247, 7.749917612, 2.751166225,
  2.329934997, 2.381489275, 1.007586167, 2.417339167,
  1.405150491, 3.70703944, 0.97220493, 1.045555418,
  0.844174113, -0.964838694, 6.706322268, -1.450891142,
  -3.72982207, -1.783092155, 0.235559123, 4.921886025,
  0.53276306, 0.971894313, -0.748116223, -1.774876193,
  -3.466883189, -0.466711247, 1.623335511, -2.038642812,
  -2.5311863, -3.439807428, -2.53471402, 0.960488749,
  -0.139743411, -0.316148543, -0.889300877, -0.288411102,
  -0.677071571, -0.827798535, 0.541419036, -0.423051443,
  0.026523786, -0.526171365, -0.583470262, -0.589116495,
  -1.373043197, -0.056887905, 0.284095465, -2.435850377,
  0.62336117, 0.754011842, 0.813193473, -0.39845453,
  -0.014356365, -0.56961647, -0.560043711, -0.011343574,
  -0.256639918, 1.844057346, 0.976336476, 0.594093031,
  -0.791484954, 0.25603043, 0.719425486, -0.374686392,
  -4.130152194, 0.873242335, 4.118420665, 1.685978684,
  -0.278632221, 0.291787716, 0.413617819, 0.498944731,
  6.305971805, -1.036058438, 5.0477785, 2.456578415,
  0.503123588, -1.152860331, -3.606945297, -4.655853367,
  -8.521704729, 0.798024875, 8.125587324, 1.371909546,
  0.91269175, 0.506320062, -3.120135304, 0.033222186,
  1.600724106, 0.160004042, 0.981493206, 2.021555304,
  1.44109419, 2.384888781, 2.456878585, -0.616605698,
  2.69071966, 3.223391183, 3.762851025, 1.270698177,
  -0.550390853, -0.447173801, 4.316196844, 2.513995405,
  3.14542727, 2.440893048, 1.781382337, 1.31207609,
  -1.811240741, -3.04137516, 9.887424622, 2.487650651,
  -0.173901336, 2.142596486, 4.617656668, 1.6376375,
  1.453901833, 1.925440542, 0.676683778, 3.141400746,
  4.577252069, 1.854424769, 2.336216689, 2.085037356,
  0.634623375, 2.657487985, 4.038683321, 3.055694741,
  2.349882722, 2.775149685, 1.821163441, 2.057919474,
  2.43355703, 0.129102095, 2.735181883, 3.394618192,
  2.183088868, 2.099635141, 1.865269704, 1.245242717,
  0.536606603, 1.073081191, 2.886129352, 3.746148414,
  0.662951984, 1.788943754
)

  
  

 
  
  
  
  
  
  
#========================
# Build hybrid model data
# ARDL(2,3), NARDL(2,1,5)
#========================

build_hybrid_data <- function(X, Y, y_lag = 2, x_lag = 5, start_lag = 6) {
  
  start_t <- start_lag + 1
  
  X_seq <- NULL
  Y_lagged <- NULL
  Target <- NULL
  
  for (t in start_t:length(Y)) {
    
    X_seq <- rbind(X_seq, X[(t - x_lag):t])
    
    Y_lagged <- rbind(Y_lagged, Y[(t - y_lag):(t - 1)])
    
    Target <- c(Target, Y[t])
  }
  
  list(
    X_seq = X_seq,
    Y_lagged = Y_lagged,
    Target = Target
  )
}

dat <- build_hybrid_data(
  X = X,
  Y = Y,
  y_lag = 2,
  x_lag = 5,
  start_lag = 6
)

X_seq <- dat$X_seq
Y_lagged <- dat$Y_lagged
Target <- dat$Target

dates_model <- dates[7:length(dates)]

#========================
# Train / Validation / Test
#========================

n <- length(Target)

#------------------------
# Train
# 2008-2022#------------------------
nTrain <- 180

#------------------------
# المتبقي بعد التدريب
#------------------------
remain <- n - nTrain

#------------------------
# تقسيم المتبقي:
# 50% Validation
# 50% Test
#------------------------
nVal <- floor(0.60 * remain)

nTest <- n - nTrain - nVal

#------------------------
# Index
#------------------------
id_train <- 1:nTrain

id_val <- (nTrain + 1):(nTrain + nVal)

id_test <- (nTrain + nVal + 1):n

#========================
# Check dates
#========================

dates_model[id_train[length(id_train)]]

dates_model[id_val[1]]

dates_model[id_test[1]]






#========================
# Scaling function
# يعتمد فقط على بيانات التدريب
#========================

scale_train <- function(train, val, test){
  
  mu  <- mean(train, na.rm = TRUE)
  sdv <- sd(train, na.rm = TRUE)
  
  if(is.na(sdv) || sdv == 0) sdv <- 1
  
  list(
    train = (train - mu) / sdv,
    val   = (val   - mu) / sdv,
    test  = (test  - mu) / sdv,
    mu    = mu,
    sd    = sdv
  )
}

#========================
# Oil scaling
#========================

X_train_sc <- X_seq[id_train, , drop = FALSE]
X_val_sc   <- X_seq[id_val, , drop = FALSE]
X_test_sc  <- X_seq[id_test, , drop = FALSE]

for(j in 1:ncol(X_seq)){
  
  tmp <- scale_train(
    train = X_seq[id_train, j],
    val   = X_seq[id_val, j],
    test  = X_seq[id_test, j]
  )
  
  X_train_sc[, j] <- tmp$train
  X_val_sc[, j]   <- tmp$val
  X_test_sc[, j]  <- tmp$test
}

#========================
# Y lag scaling
#========================

Ylag_train_sc <- Y_lagged[id_train, , drop = FALSE]
Ylag_val_sc   <- Y_lagged[id_val, , drop = FALSE]
Ylag_test_sc  <- Y_lagged[id_test, , drop = FALSE]

for(j in 1:ncol(Y_lagged)){
  
  tmp <- scale_train(
    train = Y_lagged[id_train, j],
    val   = Y_lagged[id_val, j],
    test  = Y_lagged[id_test, j]
  )
  
  Ylag_train_sc[, j] <- tmp$train
  Ylag_val_sc[, j]   <- tmp$val
  Ylag_test_sc[, j]  <- tmp$test
}

# Target scaling
Target_train=Target[id_train]
Target_val=Target[id_val]
Target_test=Target[id_test]

Tsc=scale_train(Target_train,Target_val,Target_test)

Target_train_sc=Tsc$train
Target_val_sc=Tsc$val
Target_test_sc=Tsc$test

Target_train_sc=matrix(as.numeric(Target_train_sc),ncol=1)
Target_val_sc=matrix(as.numeric(Target_val_sc),ncol=1)
Target_test_sc=matrix(as.numeric(Target_test_sc),ncol=1)

time_steps=ncol(X_seq)

X_train_arr=array(X_train_sc,dim=c(nrow(X_train_sc),time_steps,1))
X_val_arr=array(X_val_sc,dim=c(nrow(X_val_sc),time_steps,1))
X_test_arr=array(X_test_sc,dim=c(nrow(X_test_sc),time_steps,1))


#========================
# Build model function
# LSTM / CNN / HYBRID
#========================

build_model <- function(model_type,
                        filters,
                        kernel_size,
                        lstm_units,
                        dense_units,
                        drop_rate,
                        lr){
  
  k_clear_session()
  
  input_oil <- layer_input(
    shape = c(time_steps, 1),
    name  = "oil_input"
  )
  
  input_y <- layer_input(
    shape = c(ncol(Ylag_train_sc)),
    name  = "ylag_input"
  )
  
  if(model_type == "LSTM"){
    
    branch <- input_oil %>%
      layer_lstm(units = lstm_units) %>%
      layer_dropout(rate = drop_rate)
    
  } else if(model_type == "CNN"){
    
    branch <- input_oil %>%
      layer_conv_1d(
        filters     = filters,
        kernel_size = kernel_size,
        activation  = "tanh",
        padding     = "same"
      ) %>%
      layer_flatten() %>%
      layer_dropout(rate = drop_rate)
    
  } else if(model_type == "HYBRID"){
    
    branch <- input_oil %>%
      layer_conv_1d(
        filters     = filters,
        kernel_size = kernel_size,
        activation  = "tanh",
        padding     = "same"
      ) %>%
      layer_lstm(units = lstm_units) %>%
      layer_dropout(rate = drop_rate)
  }
  
  output <- layer_concatenate(list(branch, input_y)) %>%
    layer_dense(units = dense_units, activation = "tanh") %>%
    layer_dense(units = 1)
  
  model <- keras_model(
    inputs  = list(input_oil, input_y),
    outputs = output
  )
  
  model$compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss      = loss_huber()
  )
  
  return(model)
}

#========================
# Fit and evaluate model
#========================

fit_eval_model <- function(model_type,
                           filters,
                           kernel_size,
                           lstm_units,
                           dense_units,
                           drop_rate,
                           lr,
                           batch_size){
  
  filters     <- as.integer(round(filters))
  kernel_size <- as.integer(round(kernel_size))
  lstm_units  <- as.integer(round(lstm_units))
  dense_units <- as.integer(round(dense_units))
  batch_size  <- as.integer(round(batch_size))
  
  model <- build_model(
    model_type  = model_type,
    filters     = filters,
    kernel_size = kernel_size,
    lstm_units  = lstm_units,
    dense_units = dense_units,
    drop_rate   = drop_rate,
    lr          = lr
  )
  
  model$fit(
    x = list(X_train_arr, Ylag_train_sc),
    y = Target_train_sc,
    
    validation_data = list(
      list(X_val_arr, Ylag_val_sc),
      Target_val_sc
    ),
    
    epochs = as.integer(150),
    batch_size = batch_size,
    
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = as.integer(20),
        restore_best_weights = TRUE
      ),
      
      callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = as.integer(10)
      )
    ),
    
    verbose = 0
  )
  
  pred_val_sc <- as.numeric(
    model$predict(
      list(X_val_arr, Ylag_val_sc),
      verbose = 0
    )
  )
  
  pred_val <- pred_val_sc * Tsc$sd + Tsc$mu
  
  rmse <- sqrt(
    mean((Target_val - pred_val)^2, na.rm = TRUE)
  )
  
  return(list(Score = -rmse))
}

#========================
# Bayesian optimization
#========================

run_bayes <- function(model_type){
  
  BayesianOptimization(
    
    FUN = function(filters,
                   kernel_size,
                   lstm_units,
                   dense_units,
                   drop_rate,
                   lr,
                   batch_size){
      
      fit_eval_model(
        model_type  = model_type,
        filters     = filters,
        kernel_size = kernel_size,
        lstm_units  = lstm_units,
        dense_units = dense_units,
        drop_rate   = drop_rate,
        lr          = lr,
        batch_size  = batch_size
      )
    },
    
    bounds = list(
      filters     = c(8L, 32L),
      kernel_size = c(2L, 4L),
      lstm_units  = c(8L, 128L),
      dense_units = c(8L, 32L),
      drop_rate   = c(0.10, 0.50),
      lr          = c(0.0001, 0.01),
      batch_size  = c(4L, 16L)
    ),
    
    init_points = 5,
    n_iter      = 10,
    acq         = "ucb",
    kappa       = 2.576,
    verbose     = TRUE
  )
}




#========================
# Bayesian Optimization
#========================
# يتم هنا تطبيق خوارزمية Bayesian Optimization
# لاختيار أفضل القيم الفائقة (Hyperparameters)
# لكل من نماذج:
# LSTM
# CNN
# 1D-CNN-LSTM Hybrid
#
# الهدف هو تقليل قيمة RMSE على بيانات Validation
# للحصول على أفضل أداء تنبؤي للنموذج.
#========================

opt_LSTM=run_bayes("LSTM")
opt_CNN=run_bayes("CNN")
opt_HYBRID=run_bayes("HYBRID")

#========================
# Final model fitting
#========================
# يتم هنا بناء النموذج النهائي باستعمال أفضل القيم
# التي حصلنا عليها من Bayesian Optimization.
# ثم يتم تدريب النموذج على بيانات التدريب،
# ومراقبة أدائه على بيانات Validation.
# بعد ذلك يتم استخراج التنبؤات لكل من:
# Train / Validation / Test
# وإرجاعها إلى المقياس الأصلي.

final_fit <- function(model_type, opt){
  
  bp <- opt$Best_Par
  
  filters     <- as.integer(round(bp["filters"]))
  kernel_size <- as.integer(round(bp["kernel_size"]))
  lstm_units  <- as.integer(round(bp["lstm_units"]))
  dense_units <- as.integer(round(bp["dense_units"]))
  drop_rate   <- as.numeric(bp["drop_rate"])
  lr          <- as.numeric(bp["lr"])
  batch_size  <- as.integer(round(bp["batch_size"]))
  
  model <- build_model(
    model_type  = model_type,
    filters     = filters,
    kernel_size = kernel_size,
    lstm_units  = lstm_units,
    dense_units = dense_units,
    drop_rate   = drop_rate,
    lr          = lr
  )
  
  model$fit(
    x = list(X_train_arr, Ylag_train_sc),
    y = Target_train_sc,
    
    validation_data = list(
      list(X_val_arr, Ylag_val_sc),
      Target_val_sc
    ),
    
    epochs = as.integer(150),
    batch_size = batch_size,
    
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = as.integer(20),
        restore_best_weights = TRUE
      )
    ),
    
    verbose = 0
  )
  
  pred_train_sc <- as.numeric(
    model$predict(list(X_train_arr, Ylag_train_sc), verbose = 0)
  )
  
  pred_val_sc <- as.numeric(
    model$predict(list(X_val_arr, Ylag_val_sc), verbose = 0)
  )
  
  pred_test_sc <- as.numeric(
    model$predict(list(X_test_arr, Ylag_test_sc), verbose = 0)
  )
  
  pred_train <- pred_train_sc * Tsc$sd + Tsc$mu
  pred_val   <- pred_val_sc   * Tsc$sd + Tsc$mu
  pred_test  <- pred_test_sc  * Tsc$sd + Tsc$mu
  
  list(
    model      = model,
    pred_train = pred_train,
    pred_val   = pred_val,
    pred_test  = pred_test,
    best_par   = bp
  )
}


#========================
# Final fitting
#========================
# تدريب النماذج النهائية باستعمال أفضل المعلمات
# المستخرجة من Bayesian Optimization

res_LSTM   <- final_fit("LSTM", opt_LSTM)

res_CNN    <- final_fit("CNN", opt_CNN)

res_HYBRID <- final_fit("HYBRID", opt_HYBRID)


#========================
# Accuracy function
#========================
# هذه الدالة تحسب مقاييس دقة التنبؤ:
# RMSE: الجذر التربيعي لمتوسط مربعات الأخطاء
# MAE : متوسط القيم المطلقة للأخطاء
# DA  : دقة اتجاه الحركة بين القيم الفعلية والمتنبأ بها

AccFun <- function(actual, pred){
  
  RMSE <- sqrt(mean((actual - pred)^2, na.rm = TRUE))
  
  MAE <- mean(abs(actual - pred), na.rm = TRUE)
  
  
  
  DA <- mean(
    sign(diff(actual)) == sign(diff(pred)),
    na.rm = TRUE
  )
  
  c(
    RMSE = RMSE,
    MAE  = MAE,
    DA   = DA
  )
}

#========================
# ARDL and NARDL predictions aligned with hybrid sample
#========================
 
pred_ARDL  <- YHAT_ARDL[7:length(YHAT_ARDL)]

pred_NARDL <- YHAT_NARDL[7:length(YHAT_NARDL)]


#========================
# Split ARDL and NARDL predictions
#========================

pred_ARDL_train <- pred_ARDL[id_train]
pred_ARDL_val   <- pred_ARDL[id_val]
pred_ARDL_test  <- pred_ARDL[id_test]

pred_NARDL_train <- pred_NARDL[id_train]
pred_NARDL_val   <- pred_NARDL[id_val]
pred_NARDL_test  <- pred_NARDL[id_test]


#========================
# Test comparison table
#========================
# جدول مقارنة الدقة لجميع النماذج على عينة الاختبار.

Comp_Test <- rbind(
  ARDL           = AccFun(Target_test, pred_ARDL_test),
  NARDL          = AccFun(Target_test, pred_NARDL_test),
  LSTM           = AccFun(Target_test, res_LSTM$pred_test),
  CNN            = AccFun(Target_test, res_CNN$pred_test),
  `1D-CNN-LSTM`  = AccFun(Target_test, res_HYBRID$pred_test)
)

Comp_Test


#========================
# Forecast errors
#========================
# أخطاء التنبؤ على عينة الاختبار.

e_ARDL   <- Target_test - pred_ARDL_test
e_NARDL  <- Target_test - pred_NARDL_test
e_LSTM   <- Target_test - res_LSTM$pred_test
e_CNN    <- Target_test - res_CNN$pred_test
e_HYBRID <- Target_test - res_HYBRID$pred_test

#==================================================
# Diebold-Mariano Test: Hybrid vs other models
#==================================================

DM_HYBRID_ARDL  <- dm.test(e_HYBRID, e_ARDL,  h = 1, power = 2)
DM_HYBRID_NARDL <- dm.test(e_HYBRID, e_NARDL, h = 1, power = 2)
DM_HYBRID_LSTM  <- dm.test(e_HYBRID, e_LSTM,  h = 1, power = 2)
DM_HYBRID_CNN   <- dm.test(e_HYBRID, e_CNN,   h = 1, power = 2)


#==================================================
# Simple DM results table
#==================================================

DM_Table_HYBRID <- data.frame(
  Comparison = c(
    "1D-CNN-LSTM vs ARDL",
    "1D-CNN-LSTM vs NARDL",
    "1D-CNN-LSTM vs LSTM",
    "1D-CNN-LSTM vs CNN"
  ),
  
  DM_Statistic = round(c(
    as.numeric(DM_HYBRID_ARDL$statistic),
    as.numeric(DM_HYBRID_NARDL$statistic),
    as.numeric(DM_HYBRID_LSTM$statistic),
    as.numeric(DM_HYBRID_CNN$statistic)
  ), 6),
  
  P_Value = round(c(
    DM_HYBRID_ARDL$p.value,
    DM_HYBRID_NARDL$p.value,
    DM_HYBRID_LSTM$p.value,
    DM_HYBRID_CNN$p.value
  ), 6),
  
  Significant_5pct = ifelse(
    c(
      DM_HYBRID_ARDL$p.value,
      DM_HYBRID_NARDL$p.value,
      DM_HYBRID_LSTM$p.value,
      DM_HYBRID_CNN$p.value
    ) < 0.05,
    "Yes",
    "No"
  )
)


#========================
# Training sample comparison
#========================

Comp_Train <- rbind(
  ARDL = AccFun(Target_train, pred_ARDL_train),
  
  NARDL = AccFun(Target_train, pred_NARDL_train),
  
  LSTM = AccFun(Target_train, res_LSTM$pred_train),
  
  CNN = AccFun(Target_train, res_CNN$pred_train),
  
  `1D-CNN-LSTM` = AccFun(
    Target_train,
    res_HYBRID$pred_train
  )
)
 












#========================
# plot_model_panel
#========================



#========================
# Predictions
#========================

pred_ARDL_all   <- c(pred_ARDL_train, pred_ARDL_val, pred_ARDL_test)
pred_NARDL_all  <- c(pred_NARDL_train, pred_NARDL_val, pred_NARDL_test)
pred_LSTM_all   <- c(res_LSTM$pred_train, res_LSTM$pred_val, res_LSTM$pred_test)
pred_CNN_all    <- c(res_CNN$pred_train, res_CNN$pred_val, res_CNN$pred_test)
pred_HYBRID_all <- c(res_HYBRID$pred_train, res_HYBRID$pred_val, res_HYBRID$pred_test)

#========================
# Function: Full Sample
#========================

plot_model_panel <- function(model_title, pred_all, color_line, bg_fill){
  
  df_plot <- data.frame(
    Date      = dates_model,
    Actual    = Target,
    Predicted = pred_all
  )
  
  test_start <- dates_model[id_test[1]]
  
  ggplot(df_plot, aes(x = Date)) +
    
    annotate(
      "rect",
      xmin = min(df_plot$Date),
      xmax = max(df_plot$Date),
      ymin = -Inf,
      ymax = Inf,
      fill = bg_fill,
      alpha = 0.38
    ) +
    
    geom_line(
      aes(y = Actual, color = "Actual"),
      linewidth = 0.80
    ) +
    
    geom_line(
      aes(y = Predicted, color = "Predicted"),
      linewidth = 0.95
    ) +
    
    geom_vline(
      xintercept = test_start,
      color = "#E53935",
      linetype = "dashed",
      linewidth = 1.15
    ) +
    
    scale_color_manual(
      values = c(
        "Actual" = "#111111",
        "Predicted" = color_line
      ),
      labels = c(
        "Actual",
        gsub("[0-9]\\) ", "", model_title)
      )
    ) +
    
    scale_x_date(
      breaks = as.Date(c(
        "2010-01-01",
        "2015-01-01",
        "2020-01-01",
        "2025-01-01"
      )),
      labels = c("2010", "2015", "2020", "2025")
    ) +
    
    labs(
      title = model_title,
      x = "Date",
      y = "Surplus / Deficit",
      color = NULL
    ) +
    
    theme_minimal(base_size = 12) +
    
    theme(
      plot.title = element_text(
        face = "bold",
        size = 13,
        hjust = 0.5,
        color = "#0B1A6A"
      ),
      
      legend.position = "bottom",
      legend.direction = "horizontal",
      
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        color = "grey82",
        linewidth = 0.35
      ),
      
      axis.title = element_text(
        face = "bold",
        size = 11,
        color = "#111827"
      ),
      
      axis.text = element_text(
        color = "#111827",
        size = 9
      ),
      
      plot.background = element_rect(
        fill = "white",
        color = NA
      ),
      
      panel.background = element_rect(
        fill = bg_fill,
        color = "grey70"
      )
    )
}

#========================
# Full-sample figure
#========================

p_ARDL <- plot_model_panel("1) ARDL", pred_ARDL_all, "#00A896", "#E0F7F4")
p_NARDL <- plot_model_panel("2) NARDL", pred_NARDL_all, "#FF6D00", "#FFF1DD")
p_LSTM <- plot_model_panel("3) LSTM", pred_LSTM_all, "#8E24AA", "#F5E6FA")
p_CNN <- plot_model_panel("4) CNN", pred_CNN_all, "#FF2E88", "#FFE3F0")
p_HYBRID <- plot_model_panel("5) 1D-CNN-LSTM (HYBRID)", pred_HYBRID_all, "#0077FF", "#E1F0FF")

Figure_1_All_Models <-
  (p_ARDL | p_NARDL | p_LSTM) /
  (p_CNN | p_HYBRID | plot_spacer()) +
  plot_annotation(
    title = "Actual vs Predicted Values: In-sample and Out-of-sample",
    caption = "The red dashed vertical line separates the in-sample period (train + validation) from the out-of-sample test period.",
    theme = theme(
      plot.title = element_text(
        face = "bold",
        size = 23,
        hjust = 0.5,
        color = "#071A75"
      ),
      plot.caption = element_text(
        face = "italic",
        size = 11,
        hjust = 0.5,
        color = "#071A75"
      )
    )
  )

Figure_1_All_Models







LSTM_Par   <- as.data.frame(t(opt_LSTM$Best_Par))
CNN_Par    <- as.data.frame(t(opt_CNN$Best_Par))
HYBRID_Par <- as.data.frame(t(opt_HYBRID$Best_Par))

LSTM_Par$Model   <- "LSTM"
CNN_Par$Model    <- "CNN"
HYBRID_Par$Model <- "1D-CNN-LSTM"

Best_Parameters <- rbind(
  LSTM_Par,
  CNN_Par,
  HYBRID_Par
)

Best_Parameters <- Best_Parameters[
  , c("Model", setdiff(names(Best_Parameters), "Model"))
]

#=========================================
# Out-of-sample improvement relative to Hybrid
#=========================================

Test_df <- as.data.frame(Comp_Test)
Test_df$Model <- rownames(Comp_Test)

hybrid_test <- Test_df[Test_df$Model == "1D-CNN-LSTM", ]

Improvement_Test <- data.frame(
  Model = Test_df$Model,
  
  RMSE_Improvement_Percent = round(
    (Test_df$RMSE - hybrid_test$RMSE) /
      Test_df$RMSE * 100, 2
  ),
  
  MAE_Improvement_Percent = round(
    (Test_df$MAE - hybrid_test$MAE) /
      Test_df$MAE * 100, 2
  ),
  
  DA_Improvement_Percent = round(
    (hybrid_test$DA - Test_df$DA) /
      Test_df$DA * 100, 2
  )
)

# الهجين هو المرجع
Improvement_Test[
  Improvement_Test$Model == "1D-CNN-LSTM",
  -1
] <- 0

cat("\n=========================================\n")
cat("Out-of-sample improvement relative to 1D-CNN-LSTM (%)\n")
cat("=========================================\n")
print(Improvement_Test)


#========================
# Best_Parameters
#========================
print(Best_Parameters)


#=========================================================
# Note:
# Bayesian Optimization uses a common search space for all
# models. As a result, some hyperparameters may appear in
# the optimization output although they are not used in a
# particular model architecture (e.g., Filters in LSTM or
# LSTM Units in CNN).
#=========================================================


print(Comp_Train)


#========================
# Test sample comparison
#========================

print(Comp_Test)


#========================
# Diebold-Mariano Test Table
#========================

print(DM_Table_HYBRID)

#========================
#Improvement _ SAPMPEL Test
#========================


print(Improvement_Test)













save.image("Forecast_Project.RData")


