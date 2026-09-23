# with no_match
accuracy=0.9483  macro_f1=0.7074  n=18,470
per-class:
  gambling             recall=0.967  precision=0.979  f1=0.973  support=3604
  ecig                 recall=0.985  precision=0.986  f1=0.986  support=2829
  loan_shark           recall=0.952  precision=0.960  f1=0.956  support=662
  prostitution         recall=0.831  precision=0.847  f1=0.839  support=420
  hate_speech          recall=0.799  precision=0.863  f1=0.830  support=418
  pornography          recall=0.860  precision=0.862  f1=0.861  support=335
  weed                 recall=0.953  precision=0.847  f1=0.897  support=297
  kratom               recall=0.916  precision=0.939  f1=0.927  support=202
  alcohol              recall=0.853  precision=0.818  f1=0.835  support=190
  fraud                recall=0.403  precision=0.625  f1=0.490  support=62
  copyright            recall=0.652  precision=0.556  f1=0.600  support=46
  illegal_loans        recall=0.535  precision=0.575  f1=0.554  support=43
  religion             recall=0.571  precision=0.909  f1=0.702  support=35
  royal                recall=0.435  precision=0.769  f1=0.556  support=23
  illegal_labor        recall=0.429  precision=0.750  f1=0.545  support=21
  child_sexual_content recall=0.250  precision=1.000  f1=0.400  support=20
  firearms             recall=0.917  precision=1.000  f1=0.957  support=12
  forged_docs          recall=0.167  precision=1.000  f1=0.286  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.961  precision=0.948  f1=0.954  support=9243

---

# no no_match
accuracy=0.9777  macro_f1=0.8142  n=9,228
per-class:
  gambling             recall=0.991  precision=0.988  f1=0.990  support=3605
  ecig                 recall=0.996  precision=0.994  f1=0.995  support=2829
  loan_shark           recall=0.986  precision=0.978  f1=0.982  support=662
  prostitution         recall=0.931  precision=0.954  f1=0.942  support=420
  hate_speech          recall=0.962  precision=0.978  f1=0.970  support=418
  pornography          recall=0.952  precision=0.925  f1=0.938  support=335
  weed                 recall=0.987  precision=0.990  f1=0.988  support=297
  kratom               recall=0.975  precision=0.985  f1=0.980  support=202
  alcohol              recall=0.947  precision=0.909  f1=0.928  support=190
  fraud                recall=0.581  precision=0.667  f1=0.621  support=62
  copyright            recall=0.913  precision=0.913  f1=0.913  support=46
  illegal_loans        recall=0.674  precision=0.829  f1=0.744  support=43
  religion             recall=0.743  precision=0.867  f1=0.800  support=35
  royal                recall=0.783  precision=0.857  f1=0.818  support=23
  illegal_labor        recall=0.762  precision=0.889  f1=0.821  support=21
  child_sexual_content recall=0.600  precision=0.522  f1=0.558  support=20
  firearms             recall=0.917  precision=0.733  f1=0.815  support=12
  forged_docs          recall=0.667  precision=0.667  f1=0.667  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2

---

# model no no_match but data no_match
threshold คือ ถ้าค่า confident น้อยกว่าเลขที่กำหนดให้ถือว่าเป็น no_match

accuracy=0.7198  macro_f1=0.5083  n=18,470
threshold=0.75
per-class:
  gambling             recall=0.988  precision=0.675  f1=0.802  support=3604
  ecig                 recall=0.995  precision=0.817  f1=0.897  support=2829
  loan_shark           recall=0.983  precision=0.747  f1=0.849  support=662
  prostitution         recall=0.950  precision=0.657  f1=0.777  support=420
  hate_speech          recall=0.952  precision=0.422  f1=0.585  support=418
  pornography          recall=0.949  precision=0.540  f1=0.688  support=335
  weed                 recall=0.983  precision=0.745  f1=0.848  support=297
  kratom               recall=0.975  precision=0.558  f1=0.710  support=202
  alcohol              recall=0.932  precision=0.333  f1=0.490  support=190
  fraud                recall=0.516  precision=0.121  f1=0.196  support=62
  copyright            recall=0.891  precision=0.297  f1=0.446  support=46
  illegal_loans        recall=0.581  precision=0.305  f1=0.400  support=43
  religion             recall=0.914  precision=0.128  f1=0.225  support=35
  royal                recall=0.913  precision=0.131  f1=0.230  support=23
  illegal_labor        recall=0.762  precision=0.286  f1=0.416  support=21
  child_sexual_content recall=0.450  precision=0.265  f1=0.333  support=20
  firearms             recall=0.917  precision=0.224  f1=0.361  support=12
  forged_docs          recall=0.500  precision=0.200  f1=0.286  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.465  precision=0.974  f1=0.629  support=9243


accuracy=0.7375  macro_f1=0.5167  n=18,470
threshold=0.80
per-class:
  gambling             recall=0.988  precision=0.693  f1=0.815  support=3604
  ecig                 recall=0.995  precision=0.835  f1=0.908  support=2829
  loan_shark           recall=0.983  precision=0.757  f1=0.855  support=662
  prostitution         recall=0.943  precision=0.669  f1=0.783  support=420
  hate_speech          recall=0.950  precision=0.424  f1=0.586  support=418
  pornography          recall=0.943  precision=0.557  f1=0.701  support=335
  weed                 recall=0.983  precision=0.749  f1=0.850  support=297
  kratom               recall=0.975  precision=0.561  f1=0.712  support=202
  alcohol              recall=0.921  precision=0.342  f1=0.499  support=190
  fraud                recall=0.516  precision=0.130  f1=0.208  support=62
  copyright            recall=0.891  precision=0.311  f1=0.461  support=46
  illegal_loans        recall=0.558  precision=0.312  f1=0.400  support=43
  religion             recall=0.914  precision=0.134  f1=0.234  support=35
  royal                recall=0.913  precision=0.139  f1=0.241  support=23
  illegal_labor        recall=0.762  precision=0.296  f1=0.427  support=21
  child_sexual_content recall=0.400  precision=0.296  f1=0.340  support=20
  firearms             recall=0.917  precision=0.229  f1=0.367  support=12
  forged_docs          recall=0.500  precision=0.200  f1=0.286  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.501  precision=0.973  f1=0.662  support=9243


accuracy=0.7580  macro_f1=0.5336  n=18,470
threshold=0.85
per-class:
  gambling             recall=0.988  precision=0.716  f1=0.830  support=3604
  ecig                 recall=0.995  precision=0.849  f1=0.916  support=2829
  loan_shark           recall=0.980  precision=0.768  f1=0.861  support=662
  prostitution         recall=0.931  precision=0.690  f1=0.792  support=420
  hate_speech          recall=0.947  precision=0.428  f1=0.589  support=418
  pornography          recall=0.943  precision=0.575  f1=0.714  support=335
  weed                 recall=0.983  precision=0.755  f1=0.854  support=297
  kratom               recall=0.975  precision=0.563  f1=0.714  support=202
  alcohol              recall=0.921  precision=0.356  f1=0.514  support=190
  fraud                recall=0.516  precision=0.146  f1=0.228  support=62
  copyright            recall=0.891  precision=0.320  f1=0.471  support=46
  illegal_loans        recall=0.558  precision=0.364  f1=0.440  support=43
  religion             recall=0.914  precision=0.147  f1=0.253  support=35
  royal                recall=0.913  precision=0.153  f1=0.263  support=23
  illegal_labor        recall=0.762  precision=0.327  f1=0.457  support=21
  child_sexual_content recall=0.400  precision=0.381  f1=0.390  support=20
  firearms             recall=0.917  precision=0.234  f1=0.373  support=12
  forged_docs          recall=0.500  precision=0.231  f1=0.316  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.543  precision=0.971  f1=0.697  support=9243


accuracy=0.7829  macro_f1=0.5481  n=18,470
threshold=0.90
per-class:
  gambling             recall=0.987  precision=0.750  f1=0.852  support=3604
  ecig                 recall=0.994  precision=0.868  f1=0.927  support=2829
  loan_shark           recall=0.980  precision=0.782  f1=0.870  support=662
  prostitution         recall=0.924  precision=0.699  f1=0.796  support=420
  hate_speech          recall=0.945  precision=0.436  f1=0.596  support=418
  pornography          recall=0.934  precision=0.603  f1=0.733  support=335
  weed                 recall=0.980  precision=0.756  f1=0.853  support=297
  kratom               recall=0.975  precision=0.563  f1=0.714  support=202
  alcohol              recall=0.921  precision=0.377  f1=0.535  support=190
  fraud                recall=0.516  precision=0.168  f1=0.253  support=62
  copyright            recall=0.891  precision=0.333  f1=0.485  support=46
  illegal_loans        recall=0.535  precision=0.426  f1=0.474  support=43
  religion             recall=0.886  precision=0.153  f1=0.261  support=35
  royal                recall=0.913  precision=0.167  f1=0.282  support=23
  illegal_labor        recall=0.762  precision=0.356  f1=0.485  support=21
  child_sexual_content recall=0.350  precision=0.438  f1=0.389  support=20
  firearms             recall=0.917  precision=0.244  f1=0.386  support=12
  forged_docs          recall=0.500  precision=0.250  f1=0.333  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.595  precision=0.971  f1=0.738  support=9243


accuracy=0.8167  macro_f1=0.5624  n=18,470
threshold=0.95
per-class:
  gambling             recall=0.985  precision=0.793  f1=0.879  support=3604
  ecig                 recall=0.994  precision=0.889  f1=0.939  support=2829
  loan_shark           recall=0.977  precision=0.812  f1=0.887  support=662
  prostitution         recall=0.900  precision=0.759  f1=0.824  support=420
  hate_speech          recall=0.943  precision=0.447  f1=0.606  support=418
  pornography          recall=0.925  precision=0.647  f1=0.762  support=335
  weed                 recall=0.980  precision=0.768  f1=0.861  support=297
  kratom               recall=0.975  precision=0.571  f1=0.720  support=202
  alcohol              recall=0.921  precision=0.412  f1=0.569  support=190
  fraud                recall=0.484  precision=0.203  f1=0.286  support=62
  copyright            recall=0.891  precision=0.353  f1=0.506  support=46
  illegal_loans        recall=0.419  precision=0.486  f1=0.450  support=43
  religion             recall=0.857  precision=0.183  f1=0.302  support=35
  royal                recall=0.913  precision=0.223  f1=0.359  support=23
  illegal_labor        recall=0.762  precision=0.390  f1=0.516  support=21
  child_sexual_content recall=0.300  precision=0.545  f1=0.387  support=20
  firearms             recall=0.917  precision=0.262  f1=0.407  support=12
  forged_docs          recall=0.167  precision=0.250  f1=0.200  support=6
  surrogacy            recall=0.000  precision=0.000  f1=0.000  support=2
  no_match             recall=0.666  precision=0.966  f1=0.789  support=9243