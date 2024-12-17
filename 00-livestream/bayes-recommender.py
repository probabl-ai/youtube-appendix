import marimo

__generated_with = "0.9.32"
app = marimo.App(
    width="medium",
    layout_file="layouts/bayes-recommender.grid.json",
)


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import itertools as it
    return it, mo, pl


@app.cell
def __(pl):
    nrow = pl.scan_csv("~/Downloads/archive/ratings.csv").count().collect().to_dict()["userId"][0]
    nrow
    return (nrow,)


@app.cell
def __(pl):
    pl.scan_csv("~/Downloads/archive/ratings.csv").group_by("movieId").len().collect().shape
    return


@app.cell
def __(pl):
    pl.scan_csv("~/Downloads/archive/ratings.csv").group_by("userId").len().collect().shape
    return


@app.cell
def __(nrow, pl):
    batch_size = 5_000_000
    n_batch = int(nrow/batch_size)
    bcsv = pl.read_csv_batched("~/Downloads/archive/ratings.csv", batch_size=batch_size)
    return batch_size, bcsv, n_batch


@app.cell
def __(pl):
    (
        pl.scan_csv("~/Downloads/archive/ratings.csv")
        .head(10_000)
        .filter(pl.col("rating") > 3)
        .group_by("userId")
        .agg(pl.col("movieId").alias("m1"), pl.col("movieId").alias("m2"))
        .explode("m1")
        .explode("m2")
        .filter(pl.col("m1") != pl.col("m2"))
        .group_by(["m1", "m2"])
        .len()
        .sort(pl.col("len"), descending=True)
        .collect()
    )
    return


app._unparsable_cell(
    r"""
    idef calc_item_item(dataf): 
        return (
            dataf
                .filter(pl.col(\"rating\") > 3)
                .group_by(\"userId\")
                .agg(pl.col(\"movieId\").alias(\"m1\"), pl.col(\"movieId\").alias(\"m2\"))
                .explode(\"m1\")
                .explode(\"m2\")
                .group_by([\"m1\", \"m2\"])
                .len()
                .filter(pl.col(\"m1\") != pl.col(\"m2\"))
        )

    def merge_counts(*dataf): 
        if len(dataf) == 1:
            return dataf[0]
        return pl.concat(*dataf).group_by([\"m1\", \"m2\"]).count()
    """,
    name="__"
)


@app.cell
def __():
    import time
    return (time,)


@app.cell
def __(bcsv, calc_item_item, n_batch, pl, time):
    df_accum = bcsv.next_batches(1)[0].pipe(calc_item_item)
    for i in range(n_batch - 1): 
        tic = time.time()
        next_batch = bcsv.next_batches(1)[0].pipe(calc_item_item)
        df_accum = pl.concat([df_accum, next_batch]).group_by(["m1", "m2"]).sum()
        toc = time.time()
        print(f"batch {i}/{n_batch}: took {toc - tic: .2f}s shape={df_accum.shape}")
        print(df_accum.max())
    return df_accum, i, next_batch, tic, toc


@app.cell
def __(df_accum):
    df_accum.write_parquet("item_item.parquet")
    return


@app.cell
def __(df_accum):
    df_accum.sort("len", descending=True).head()
    return


@app.cell
def __(pl):
    movie_popularity_df = (
        pl.scan_csv("~/Downloads/archive/ratings.csv")
            .group_by("movieId")
            .len()
            .collect()
            .select(movieId=pl.col("movieId"), freq=pl.col("len"))
    )
    return (movie_popularity_df,)


@app.cell
def __(df_accum, pl):
    seen_movie = 1188

    df_accum.filter(pl.col("m1") == seen_movie).sort(pl.col("len"), descending=True)
    return (seen_movie,)


@app.cell
def __(pl):
    name_df = pl.read_csv("~/Downloads/archive/movies_metadata.csv", infer_schema_length=100_000).select(pl.col("id").alias("movieId"), "original_title")
    return (name_df,)


@app.cell
def __(mo):
    text_input = mo.ui.text(label="Search movie")
    return (text_input,)


@app.cell
def __(mo):
    get_state, set_state = mo.state(value=[])
    return get_state, set_state


@app.cell
def __():
    import numpy as np
    return (np,)


@app.cell
def __(mo, np, set_state):
    reset_btn = mo.ui.button(label="Reset", on_change=lambda d: set_state([]))
    smoothing_slider = mo.ui.slider(label="smoothing", steps=np.logspace(-2, 0), show_value=True)
    return reset_btn, smoothing_slider


@app.cell
def __(
    get_state,
    mo,
    name_df,
    pl,
    reset_btn,
    set_state,
    smoothing_slider,
    text_input,
):
    movies = (
        name_df
          .filter(pl.col("original_title").str.to_lowercase().str.contains(text_input.value))
          .head(10)
          .to_dicts()
    )

    def updater(movie_id): 
        def update(checked): 
            if checked:
                set_state(get_state() + [movie_id])
            else:
                set_state([s for s in get_state() if s != movie_id])
        return update

    array = mo.ui.array([
        mo.ui.checkbox(
            label=m["original_title"], 
            on_change=updater(m["movieId"])
        ) for m in movies
    ])

    mo.hstack([
        mo.vstack([
            reset_btn, 
            smoothing_slider, 
            text_input, 
        ]),
        array
    ])
    return array, movies, updater


@app.cell
def __(get_state):
    get_state()
    return


@app.cell(hide_code=True)
def __(
    df_accum,
    get_state,
    movie_popularity_df,
    name_df,
    pl,
    smoothing_slider,
):
    selection = [int(i) for i in get_state()]

    (
        df_accum
        .filter(pl.col("m1").is_in(selection))
        .group_by(["m1", "m2"])
        .sum()
        .select(pl.col("m2").cast(pl.String), pl.col("len").alias("p_together"))
        .join(name_df, left_on="m2", right_on="movieId")
        .with_columns(pl.col("m2").cast(pl.Int64).alias("movieId"))
        .drop("m2")
        .select("movieId", "original_title", "p_together")
        .filter(pl.col("p_together") > 2)
        .join(movie_popularity_df, left_on="movieId", right_on="movieId")
        .with_columns(score=pl.col("p_together")/pl.col("freq")**smoothing_slider.value)
        .sort(pl.col("score"), descending=True)
        .head(20)
    )
    return (selection,)


@app.cell
def __(mo):
    mo.md(
        """
        # Huh?

        We can explain how this algorithm should work in theory, but you might notices that it does not work super well in practice. Why might that be? 

        To understand, we will need to check the data itself.
        """
    )
    return


@app.cell
def __(mo, pl):
    ratings_in = pl.scan_csv("~/Downloads/archive/ratings.csv")

    mo.hstack([
        ratings_in.group_by("rating").len().collect().plot.line("rating","len").properties(title="distribution of ratings"),
        ratings_in.group_by("userId").agg(pl.col("rating").len().alias("n"), pl.col("rating").mean().alias("mean_rating")).collect().group_by("n").mean().plot.scatter("n","mean_rating").properties(title="distribution of ratings per user popularity"),
        ratings_in.group_by("movieId").agg(pl.col("rating").len().alias("n"), pl.col("rating").mean().alias("mean_rating")).collect().group_by("n").mean().plot.scatter("n","mean_rating").properties(title="distribution of ratings per movie popularity")
    ])
    return (ratings_in,)


@app.cell
def __(mo):
    mo.md(
        """
        Another important aspect of this is that we are taking ratings from a user over a potentially long timespan. 

        Maybe all of this would be better if we did some sessions first? Where we user might be defined by the year that they are in as well. This will obviously have all sorts of consequences, but the dataset spans more than a decade!
        """
    )
    return


@app.cell
def __(pl, ratings_in):
    (
        ratings_in
            .select((pl.col("timestamp")*1_000_000).cast(pl.Datetime))
            .select(
                pl.col("timestamp").min().alias("min_ts"), 
                pl.col("timestamp").max().alias("max_ts")
            ).collect()
    )
    return


@app.cell
def __(mo):
    mo.md("""To get a bit more of an impression of a single user, lets explore this.""")
    return


@app.cell
def __(name_df, pl):
    df_orig = (
        pl.scan_csv("~/Downloads/archive/ratings.csv")
    )

    (
        df_orig
            .filter(pl.col("userId") == 2060)
            .collect()
            .filter(pl.col("rating") > 3)
            .with_columns(pl.col("movieId").cast(pl.String))
            .join(name_df, right_on="movieId", left_on="movieId")
            .select("original_title", "rating")
    )
    return (df_orig,)


@app.cell
def __(mo):
    mo.md(
        """
        Another thing to keep in mind is that we are also throwing away a bunch of information. Bad ratings also tell us something. 

        ### Then again ...

        There is a reason why folks tend to ignore rating-based features in recommenders these days. Instead of asking for ratings, it may be less biased to observe behavior instead and to base recommendations on clicks/watch time. 

        ### Welcome to recommender-land! 

        Stuff can take a while to get right and a lot of it revolves around understanding your platform/users!
        """
    )
    return

if __name__ == "__main__":
    app.run()
