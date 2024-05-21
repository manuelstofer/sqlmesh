from __future__ import annotations

import typing as t
import sys

import pandas as pd
from sqlglot import exp

from sqlmesh.core.dialect import to_schema, transform_values
from sqlmesh.core.engine_adapter.base_postgres import BasePostgresEngineAdapter
from sqlmesh.core.engine_adapter.mixins import (
    GetCurrentCatalogFromFunctionMixin,
    LogicalMergeMixin,
    NonTransactionalTruncateMixin,
)
from sqlmesh.core.engine_adapter.shared import (
    CommentCreationView,
    DataObject,
    DataObjectType,
    SourceQuery,
    set_catalog,
)
from sqlmesh.utils import random_id

if t.TYPE_CHECKING:
    from sqlmesh.core._typing import SchemaName, TableName
    from sqlmesh.core.engine_adapter.base import QueryOrDF


@set_catalog()
class RedshiftEngineAdapter(
    BasePostgresEngineAdapter,
    LogicalMergeMixin,
    GetCurrentCatalogFromFunctionMixin,
    NonTransactionalTruncateMixin,
):
    DIALECT = "redshift"
    CURRENT_CATALOG_EXPRESSION = exp.func("current_database")
    # Redshift doesn't support comments for VIEWs WITH NO SCHEMA BINDING (which we always use)
    COMMENT_CREATION_VIEW = CommentCreationView.UNSUPPORTED
    SUPPORTS_REPLACE_TABLE = False

    # The DEFAULT_BATCH_SIZE inherited from Postgres seems too small and slows down DataFrame imports
    # Resetting it back to the default value
    DEFAULT_BATCH_SIZE = 10000

    def _columns_query(self, table: exp.Table) -> exp.Select:
        sql = (
            exp.select("column_name", "data_type")
            .from_("svv_columns")  # Includes late-binding views
            .where(exp.column("table_name").eq(table.alias_or_name))
        )
        if table.args.get("db"):
            sql = sql.where(exp.column("table_schema").eq(table.args["db"].name))
        return sql

    @property
    def cursor(self) -> t.Any:
        # Redshift by default uses a `format` paramstyle that has issues when we try to write our snapshot
        # data to snapshot table. There doesn't seem to be a way to disable parameter overriding so we just
        # set it to `qmark` since that doesn't cause issues.
        cursor = self._connection_pool.get_cursor()
        cursor.paramstyle = "qmark"
        return cursor

    def _fetch_native_df(
        self, query: t.Union[exp.Expression, str], quote_identifiers: bool = False
    ) -> pd.DataFrame:
        """Fetches a Pandas DataFrame from the cursor"""
        self.execute(query, quote_identifiers=quote_identifiers)
        return self.cursor.fetch_dataframe()

    def _create_table_from_source_queries(
        self,
        table_name: TableName,
        source_queries: t.List[SourceQuery],
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        exists: bool = True,
        replace: bool = False,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Redshift doesn't support `CREATE TABLE IF NOT EXISTS AS...` but does support `CREATE TABLE AS...` so
        we check if the exists check exists and if not then we can use the base implementation. Otherwise we
        manually check if it exists and if it does then this is a no-op anyways so we return and if it doesn't
        then we run the query with exists set to False since we just confirmed it doesn't exist.
        """
        if not exists:
            return super()._create_table_from_source_queries(
                table_name,
                source_queries,
                columns_to_types,
                exists,
                table_description=table_description,
                column_descriptions=column_descriptions,
                **kwargs,
            )
        if self.table_exists(table_name):
            return
        super()._create_table_from_source_queries(
            table_name,
            source_queries,
            exists=False,
            table_description=table_description,
            column_descriptions=column_descriptions,
            **kwargs,
        )

    def _build_create_table_exp(
        self,
        table_name_or_schema: t.Union[exp.Schema, TableName],
        expression: t.Optional[exp.Expression],
        exists: bool = True,
        replace: bool = False,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        table_description: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> exp.Create:
        statement = super()._build_create_table_exp(
            table_name_or_schema,
            expression=expression,
            exists=exists,
            replace=replace,
            columns_to_types=columns_to_types,
            table_description=table_description,
            **kwargs,
        )

        if (
            statement.expression
            and statement.expression.args.get("limit") is not None
            and statement.expression.args["limit"].expression.this == "0"
        ):
            assert not isinstance(table_name_or_schema, exp.Schema)
            # redshift has a bug where CTAS statements have non determistic types. if a limit
            # is applied to a ctas statement, VARCHAR types default to 1 in some instances.
            # this checks the explain plain from redshift and tries to detect when these optimizer
            # bugs occur and force a cast
            select_statement = statement.expression.copy()
            for select_or_union in select_statement.find_all(exp.Select, exp.Union):
                select_or_union.set("limit", None)
                select_or_union.set("where", None)

            temp_view_name = exp.table_(f"#sqlmesh__{random_id()}")
            self.create_view(
                temp_view_name, select_statement, replace=False, no_schema_binding=False
            )
            columns_to_types_from_view = self.columns(temp_view_name)

            schema = self._build_schema_exp(
                exp.to_table(table_name_or_schema),
                columns_to_types_from_view,
            )
            statement = super()._build_create_table_exp(
                schema,
                None,
                exists=exists,
                replace=replace,
                columns_to_types=columns_to_types_from_view,
                table_description=table_description,
                **kwargs,
            )

        return statement

    def create_view(
        self,
        view_name: TableName,
        query_or_df: QueryOrDF,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        replace: bool = True,
        materialized: bool = False,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        view_properties: t.Optional[t.Dict[str, exp.Expression]] = None,
        **create_kwargs: t.Any,
    ) -> None:
        """
        Redshift views are "binding" by default to their underlying table which means you can't drop that
        underlying table without dropping the view first. This is a problem for us since we want to be able to
        swap tables out from under views. Therefore, we create the view as non-binding.
        """
        return super().create_view(
            view_name,
            query_or_df,
            columns_to_types,
            replace,
            materialized,
            table_description=table_description,
            column_descriptions=column_descriptions,
            no_schema_binding=create_kwargs.pop("no_schema_binding", True),
            view_properties=view_properties,
            **create_kwargs,
        )

    def replace_query(
        self,
        table_name: TableName,
        query_or_df: QueryOrDF,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Redshift doesn't support `CREATE OR REPLACE TABLE...` and it only supports `VALUES` in
        `INSERT` expressions, so we need to specially handle DataFrame replacements.

        DataFrames are loaded to Redshift using multi-row inserts as `INSERT INTO... VALUES (...), (...)`.
        This method does not allow us to use the implementation with SourceQuery, but it is much
        faster than using (`SELECT`... `UNION` ...) to replace the missing `VALUES` support in
        `SELECT` expressions.

        If the table doesn't exist then we just create it and load it with insert statements
        If it does exist then we need to do the:
            `CREATE TABLE...`, `INSERT INTO...`, `RENAME TABLE...`, `RENAME TABLE...`, DROP TABLE...`  dance.
        """
        if not self.is_pandas_df(query_or_df) or not self.table_exists(table_name):
            return super().replace_query(
                table_name,
                query_or_df,
                columns_to_types,
                table_description,
                column_descriptions,
                **kwargs,
            )

        columns_to_types = self._columns_to_types(query_or_df, columns_to_types)
        target_table = exp.to_table(table_name)
        df = query_or_df[list(columns_to_types)]

        with self.transaction():
            temp_table = self._get_temp_table(target_table)
            old_table = self._get_temp_table(target_table)
            self.create_table(
                temp_table,
                columns_to_types,
                exists=False,
                table_description=table_description,
                column_descriptions=column_descriptions,
                **kwargs,
            )
            self._insert_append_multi_row(temp_table, df, columns_to_types)
            self.rename_table(target_table, old_table)
            self.rename_table(temp_table, target_table)
            self.drop_table(old_table)

    def _insert_append_multi_row(
        self,
        table_name: TableName,
        df: pd.DataFrame,
        columns_to_types: t.Dict[str, exp.DataType],
        batch_size: t.Optional[int] = None,
    ) -> None:
        """
        Inserts the given DataFrame into the given table using multi-row inserts
        """
        batch_size = self.DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        for values_exp in self._values_exp_for_batch_range(df, batch_size, columns_to_types):
            insert_sql = exp.insert(values_exp, table_name, columns=list(columns_to_types))
            self.execute(insert_sql)

    def _values_exp_for_batch_range(
        self,
        df: pd.DataFrame,
        batch_size: int,
        columns_to_types: t.Dict[str, exp.DataType],
    ) -> t.Iterator[t.List[exp.Select]]:
        """
        Returns a generator that batches the DataFrame into `VALUES` expressions
        """
        assert isinstance(df, pd.DataFrame)
        num_rows = len(df.index)
        batch_size = sys.maxsize if batch_size == 0 else batch_size
        rows = list(df.itertuples(index=False, name=None))
        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(num_rows, batch_start + batch_size)
            batch = [
                tuple(transform_values(row, columns_to_types))
                for row in rows[batch_start:batch_end]
            ]
            yield exp.values(batch)

    def _get_data_objects(
        self, schema_name: SchemaName, object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """
        Returns all the data objects that exist in the given schema and optionally catalog.
        """
        catalog = self.get_current_catalog()
        table_query = exp.select(
            exp.column("schemaname").as_("schema_name"),
            exp.column("tablename").as_("name"),
            exp.Literal.string("TABLE").as_("type"),
        ).from_("pg_tables")
        view_query = (
            exp.select(
                exp.column("schemaname").as_("schema_name"),
                exp.column("viewname").as_("name"),
                exp.Literal.string("VIEW").as_("type"),
            )
            .from_("pg_views")
            .where(exp.column("definition").ilike("%create materialized view%").not_())
        )
        materialized_view_query = (
            exp.select(
                exp.column("schemaname").as_("schema_name"),
                exp.column("viewname").as_("name"),
                exp.Literal.string("MATERIALIZED_VIEW").as_("type"),
            )
            .from_("pg_views")
            .where(exp.column("definition").ilike("%create materialized view%"))
        )
        subquery = exp.union(
            table_query,
            exp.union(view_query, materialized_view_query, distinct=False),
            distinct=False,
        )
        query = (
            exp.select("*")
            .from_(subquery.subquery(alias="objs"))
            .where(exp.column("schema_name").eq(to_schema(schema_name).db))
        )
        if object_names:
            query = query.where(exp.column("name").isin(*object_names))
        df = self.fetchdf(query)
        return [
            DataObject(
                catalog=catalog,
                schema=row.schema_name,
                name=row.name,
                type=DataObjectType.from_str(row.type),  # type: ignore
            )
            for row in df.itertuples()
        ]
