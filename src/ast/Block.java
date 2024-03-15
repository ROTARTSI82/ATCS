package ast;

import java.util.ArrayList;

public class Block implements Statement
{
    private final ArrayList<Statement> sub = new ArrayList<>();

    public void add(Statement stmt)
    {
        sub.add(stmt);
    }

    @Override
    public void exec(Environment env)
    {
        for (Statement state: sub)
            state.exec(env);
    }
}
