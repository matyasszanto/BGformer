# SIformer

Add the following code to _base_model.py after `super.__init__()`

```
# remove num_workers_loader from trainer_kwargs
if trainer_kwargs.get("num_workers_loader", None) is not None:
    trainer_kwargs = {k:v for k,v in trainer_kwargs.items() if k != "num_workers_loader"}
```

Reason: 
newer versions of Pytorch Lightning models don't take num_workers_loader as a keyword argument.

